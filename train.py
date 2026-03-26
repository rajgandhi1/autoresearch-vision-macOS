"""
CLIP-style vision-language pretraining script. Single-GPU/MPS, single-file.
Cherry-picked and adapted from the autoresearch text pretraining setup.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()


verify_macos_env()

from prepare import IMAGE_SIZE, PATCH_SIZE, MAX_TEXT_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_recall

# ---------------------------------------------------------------------------
# Shared transformer components
# ---------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd,      bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidirectional
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1  = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2  = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = SelfAttention(n_embd, n_head)
        self.mlp  = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer (ViT image encoder)
# ---------------------------------------------------------------------------

@dataclass
class ViTConfig:
    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    n_layer:    int = 4
    n_head:     int = 4
    n_embd:     int = 256
    embed_dim:  int = 128


class PatchEmbed(nn.Module):
    """Split image into patches and project to n_embd. Prepends a CLS token."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        n_patches    = (config.image_size // config.patch_size) ** 2
        self.proj    = nn.Conv2d(3, config.n_embd, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, config.n_embd))

    def forward(self, x):
        B   = x.shape[0]
        x   = self.proj(x).flatten(2).transpose(1, 2)       # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                    # [B, N+1, C]
        return x + self.pos_embed


class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.blocks      = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.out_proj    = nn.Linear(config.n_embd, config.embed_dim, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(norm(x[:, 0]))  # CLS token at position 0


# ---------------------------------------------------------------------------
# Text Transformer (text encoder)
# ---------------------------------------------------------------------------

@dataclass
class TextConfig:
    vocab_size:  int = 50257        # GPT-2 vocab (tiktoken gpt2)
    max_seq_len: int = MAX_TEXT_LEN
    n_layer:     int = 4
    n_head:      int = 4
    n_embd:      int = 256
    embed_dim:   int = 128


class TextTransformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.wte      = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe      = nn.Embedding(config.max_seq_len, config.n_embd)
        self.blocks   = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.out_proj = nn.Linear(config.n_embd, config.embed_dim, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        h    = self.wte(x) + self.wpe(pos)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(norm(h[:, 0]))  # CLS token prepended at position 0 by Tokenizer


# ---------------------------------------------------------------------------
# CLIP Model
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    def __init__(self, vit_config: ViTConfig, text_config: TextConfig):
        super().__init__()
        assert vit_config.embed_dim == text_config.embed_dim, "embed_dim must match between encoders"
        self.vision      = VisionTransformer(vit_config)
        self.text        = TextTransformer(text_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    @torch.no_grad()
    def init_weights(self):
        # Patch embedding
        nn.init.normal_(self.vision.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.vision.patch_embed.cls_token)
        nn.init.normal_(self.vision.patch_embed.pos_embed, std=0.02)
        # Text embeddings
        nn.init.normal_(self.text.wte.weight, std=0.02)
        nn.init.normal_(self.text.wpe.weight, std=0.02)
        # Transformer blocks: zero projection weights for stable early training
        for blocks in [self.vision.blocks, self.text.blocks]:
            for block in blocks:
                nn.init.normal_(block.attn.qkv.weight,  std=0.02)
                nn.init.zeros_(block.attn.proj.weight)
                nn.init.normal_(block.mlp.fc1.weight,   std=0.02)
                nn.init.zeros_(block.mlp.fc2.weight)
        # Output projections
        nn.init.normal_(self.vision.out_proj.weight, std=0.02)
        nn.init.normal_(self.text.out_proj.weight,   std=0.02)

    def encode(self, images, texts):
        """Return L2-normalised image and text feature vectors."""
        img_feat = F.normalize(self.vision(images), dim=-1)
        txt_feat = F.normalize(self.text(texts),    dim=-1)
        return img_feat, txt_feat

    def forward(self, images, texts):
        """Symmetric InfoNCE (CLIP) contrastive loss."""
        img_feat, txt_feat = self.encode(images, texts)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * img_feat @ txt_feat.T        # [B, B]
        B      = images.shape[0]
        labels = torch.arange(B, device=images.device)
        loss   = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizer(self, matrix_lr, embedding_lr, weight_decay, adam_betas):
        """Route nn.Linear weights to Muon, everything else (embeddings, conv, scalars) to AdamW."""
        linear_ids    = {id(m.weight) for m in self.modules() if isinstance(m, nn.Linear)}
        matrix_params = [p for p in self.parameters() if id(p) in linear_ids]
        other_params  = [p for p in self.parameters() if id(p) not in linear_ids]

        param_groups = [
            dict(kind='adamw', params=other_params, lr=embedding_lr,
                 betas=adam_betas, eps=1e-8, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461,   -22.48329292557795,   15.878769915207462),
    (4.042929935166739,   -2.808917465908714,    0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,    0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,   0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,   0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    step_t  = step_t.to(device=p.device,  dtype=p.dtype)
    lr_t    = lr_t.to(device=p.device,    dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t   = eps_t.to(device=p.device,   dtype=p.dtype)
    wd_t    = wd_t.to(device=p.device,    dtype=p.dtype)

    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t       = lr_t.to(device=stacked_params.device,       dtype=stacked_params.dtype)
    wd_t       = wd_t.to(device=stacked_params.device,       dtype=stacked_params.dtype)
    beta2_t    = beta2_t.to(device=stacked_params.device,    dtype=stacked_params.dtype)

    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    v_mean     = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm     = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2_cast)
    step_size  = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr   = lr_t.to(g.dtype)
    wd   = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for Linear weight matrices, AdamW for everything else."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        # torch.compile is unstable on MPS — only use on CUDA/CPU
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type in ("cuda", "cpu"):
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused  = torch.compile(muon_step_fused,  **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused  = muon_step_fused

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step']        = 0
                state['exp_avg']     = torch.zeros_like(p)
                state['exp_avg_sq']  = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            self.adamw_step_fused(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p     = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim        = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads  = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
            self._muon_beta2_t, group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — this is the file the agent modifies)
# ---------------------------------------------------------------------------

# Vision encoder
VIT_DEPTH = 4           # number of ViT transformer layers
VIT_DIM   = 256         # ViT embedding dimension

# Text encoder
TEXT_DEPTH = 4          # number of text transformer layers
TEXT_DIM   = 256        # text embedding dimension

# Shared
EMBED_DIM = 128         # contrastive projection dimension (both encoders must match)
HEAD_DIM  = 64          # target attention head dimension (n_head = dim // HEAD_DIM)

# Optimization
DEVICE_BATCH_SIZE = 64  # image-text pairs per forward pass (reduce if OOM)
TOTAL_BATCH_SIZE  = 64  # effective batch size (must be multiple of DEVICE_BATCH_SIZE)
EMBEDDING_LR  = 3e-4    # AdamW LR for embeddings, conv weights, scalars
MATRIX_LR     = 0.01    # Muon LR for Linear weight matrices
WEIGHT_DECAY  = 0.1     # Muon weight decay (applied to matrix params only)
ADAM_BETAS    = (0.9, 0.95)
WARMUP_RATIO   = 0.1    # fraction of TIME_BUDGET for LR warmup
WARMDOWN_RATIO = 0.4    # fraction of TIME_BUDGET for LR cooldown
FINAL_LR_FRAC  = 0.0    # final LR as fraction of peak

# ---------------------------------------------------------------------------
# Setup: device, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device      = torch.device(device_type)

if device_type == "cuda":
    torch.cuda.manual_seed(42)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device_type == "cpu":
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:  # mps — bfloat16 autocast not supported
    import contextlib
    autocast_ctx = contextlib.nullcontext()

tokenizer = Tokenizer()


def build_configs():
    vit_heads  = max(1, VIT_DIM  // HEAD_DIM)
    text_heads = max(1, TEXT_DIM // HEAD_DIM)
    vit_config  = ViTConfig(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
        n_layer=VIT_DEPTH,  n_head=vit_heads,  n_embd=VIT_DIM,  embed_dim=EMBED_DIM,
    )
    text_config = TextConfig(
        vocab_size=tokenizer.get_vocab_size(), max_seq_len=MAX_TEXT_LEN,
        n_layer=TEXT_DEPTH, n_head=text_heads, n_embd=TEXT_DIM, embed_dim=EMBED_DIM,
    )
    return vit_config, text_config


vit_config, text_config = build_configs()
print(f"ViT config:  {asdict(vit_config)}")
print(f"Text config: {asdict(text_config)}")

model = CLIP(vit_config, text_config).to(device)
model.init_weights()

num_params = model.num_params()
print(f"Total parameters: {num_params / 1e6:.1f}M")

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

optimizer = model.setup_optimizer(
    matrix_lr=MATRIX_LR,
    embedding_lr=EMBEDDING_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
)

# torch.compile is unstable on MPS — only enable on CUDA
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

train_loader         = make_dataloader(DEVICE_BATCH_SIZE, "train")
images, texts        = next(train_loader)  # prefetch first batch

print(f"Time budget:  {TIME_BUDGET}s")
print(f"Batch size:   {TOTAL_BATCH_SIZE} (grad_accum={grad_accum_steps})")

# ---------------------------------------------------------------------------
# LR and momentum schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


t_start_training   = time.time()
smooth_loss        = 0.0
total_training_time = 0.0
step               = 0

while True:
    sync_device(device_type)
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(images, texts)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        images, texts = next(train_loader)

    # Schedules
    progress       = min(total_training_time / TIME_BUDGET, 1.0)
    lrm            = get_lr_multiplier(progress)
    muon_momentum  = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum

    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss explodes (should be ~log(batch_size) at init)
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # EMA loss logging
    ema_beta   = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss_f
    debiased    = smooth_loss / (1 - ema_beta ** (step + 1))
    pct_done    = 100 * progress
    remaining   = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.4f} | lrm: {lrm:.2f} "
        f"| dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # GC management (Python GC can cause stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stop after TIME_BUDGET, but skip the warmup steps (compilation overhead)
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

total_images = step * TOTAL_BATCH_SIZE

# Final evaluation
model.eval()
with autocast_ctx:
    metrics = evaluate_recall(model, DEVICE_BATCH_SIZE)

t_end = time.time()
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_recall@1:     {metrics['recall@1']:.6f}")
print(f"val_recall@5:     {metrics['recall@5']:.6f}")
print(f"val_recall@10:    {metrics['recall@10']:.6f}")
print(f"val_mAP:          {metrics['mAP']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_images_M:   {total_images / 1e6:.2f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"vit_depth:        {VIT_DEPTH}")
print(f"text_depth:       {TEXT_DEPTH}")
