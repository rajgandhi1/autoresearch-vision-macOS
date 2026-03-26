"""
One-time data preparation for vision-language autoresearch experiments.
Downloads a subset of WikiArt from HuggingFace and caches images + metadata.

Usage:
    uv run prepare.py                          # default (3500 train + 500 val)
    uv run prepare.py --num-train 500          # smaller subset for testing

Data cached at ~/.cache/autoresearch/wikiart/.
"""

import os
import sys
import time
import math
import json
import random
import argparse

import torch
import tiktoken
from PIL import Image
import torchvision.transforms as T


def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()


verify_macos_env()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE   = 128    # input image resolution (H × W)
PATCH_SIZE   = 16     # ViT patch size → (IMAGE_SIZE / PATCH_SIZE)² patches per image
MAX_TEXT_LEN = 32     # max tokens per text description (including CLS token)
TIME_BUDGET  = 300    # training time budget in seconds (5 minutes)
EVAL_IMAGES  = 256    # validation images used for recall@1 evaluation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR      = os.path.join(CACHE_DIR, "wikiart")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

NUM_TRAIN = 3500
NUM_VAL   = 500

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    GPT-2 tokenizer via tiktoken. No training required.
    The EOT token (50256) is prepended as a CLS token at position 0.
    TextTransformer reads position 0 as the text representation.
    """

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.cls_token = self.enc.eot_token  # 50256

    @classmethod
    def from_directory(cls):
        return cls()

    def get_vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text):
        """Return a fixed-length list of MAX_TEXT_LEN token ids. CLS at position 0."""
        tokens = [self.cls_token]
        tokens += self.enc.encode_ordinary(text)[: MAX_TEXT_LEN - 1]
        tokens = tokens[:MAX_TEXT_LEN]
        tokens += [0] * (MAX_TEXT_LEN - len(tokens))  # zero-pad
        return tokens


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def make_text(sample):
    """Construct text description from WikiArt metadata fields."""
    artist = str(sample.get("artist") or "Unknown")
    style  = str(sample.get("style")  or "Unknown")
    return f"A {style} painting by {artist}"


def prepare_data(num_train=NUM_TRAIN, num_val=NUM_VAL):
    """Stream WikiArt from HuggingFace and cache images + metadata to disk."""
    if os.path.exists(METADATA_FILE):
        print(f"Data: already prepared at {DATA_DIR}")
        return

    print("Data: downloading WikiArt subset from HuggingFace (streaming)...")
    from datasets import load_dataset

    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val"),   exist_ok=True)

    dataset = load_dataset("huggan/wikiart", split="train", streaming=True, trust_remote_code=True)

    metadata = {"train": [], "val": []}
    total = num_train + num_val
    t0 = time.time()

    for i, sample in enumerate(dataset):
        if i >= total:
            break
        split    = "train" if i < num_train else "val"
        img_path = os.path.join(DATA_DIR, split, f"{i:06d}.jpg")
        try:
            img  = sample["image"].convert("RGB")
            img.save(img_path, quality=85)
            text = make_text(sample)
            metadata[split].append({"path": img_path, "text": text})
        except Exception as e:
            print(f"  Skipping sample {i}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} images cached ({time.time() - t0:.0f}s elapsed)")

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"Data: {len(metadata['train'])} train + {len(metadata['val'])} val images saved to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

_VAL_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


def make_dataloader(batch_size, split):
    """
    Infinite dataloader for image-text pairs.
    Yields (images [B, 3, H, W], text_tokens [B, T]) on device.
    Shuffles the dataset each epoch.
    """
    assert split in ("train", "val")

    with open(METADATA_FILE) as f:
        metadata = json.load(f)[split]

    tokenizer = Tokenizer()
    device    = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    transform = _TRAIN_TRANSFORM if split == "train" else _VAL_TRANSFORM

    indices = list(range(len(metadata)))

    while True:
        random.shuffle(indices)
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch  = [metadata[j] for j in indices[i : i + batch_size]]
            images, texts = [], []
            for item in batch:
                img = Image.open(item["path"]).convert("RGB")
                images.append(transform(img))
                texts.append(torch.tensor(tokenizer.encode(item["text"]), dtype=torch.long))
            yield torch.stack(images).to(device), torch.stack(texts).to(device)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — these are the fixed metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_recall(model, batch_size):
    """
    Image-to-text retrieval metrics on EVAL_IMAGES validation samples.
    Returns recall@1, recall@5, recall@10, and mAP.
    - recall@k: fraction of image queries whose correct text is in the top-k results.
    - mAP: mean reciprocal rank (= mAP when each query has one relevant item).
    All metrics range 0.0 to 1.0; higher is better.
    """
    val_loader = make_dataloader(batch_size, "val")
    steps = EVAL_IMAGES // batch_size

    all_img, all_txt = [], []
    for _ in range(steps):
        images, texts = next(val_loader)
        img_feat, txt_feat = model.encode(images, texts)
        all_img.append(img_feat.cpu().float())
        all_txt.append(txt_feat.cpu().float())

    img_feats = torch.cat(all_img)   # [N, embed_dim]
    txt_feats = torch.cat(all_txt)   # [N, embed_dim]
    sim = img_feats @ txt_feats.T    # [N, N] cosine similarity (features already normalized)
    n   = img_feats.shape[0]

    # 0-indexed rank of the correct text for each image query
    sorted_idx    = sim.argsort(dim=1, descending=True)                          # [N, N]
    correct_ranks = (sorted_idx == torch.arange(n).unsqueeze(1)).nonzero()[:, 1] # [N]

    return {
        "recall@1":  (correct_ranks < 1).float().mean().item(),
        "recall@5":  (correct_ranks < 5).float().mean().item(),
        "recall@10": (correct_ranks < 10).float().mean().item(),
        "mAP":       (1.0 / (correct_ranks.float() + 1)).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WikiArt data for vision-language autoresearch")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN, help="Training images to cache")
    parser.add_argument("--num-val",   type=int, default=NUM_VAL,   help="Validation images to cache")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()
    prepare_data(args.num_train, args.num_val)
    print()
    print("Done! Ready to train.")
