import os
import torch
from pathlib import Path
from model import GPT, GPTConfig
from tokenizers import ByteLevelBPETokenizer
from safetensors.torch import load_file, save_file
import json
import hashlib

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "model.pt"
SAFETENSOR_PATH = BASE_DIR / "model.safetensors"
VOCAB_PATH = BASE_DIR / "vocab.json"
MERGES_PATH = BASE_DIR / "merges.txt"
SOUL_STATE_PATH = BASE_DIR / "Luma_EvotypeIII.json"
SOUL_DATA_PATH = BASE_DIR / "soul_reflections.json"

# Tokenizer
tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
encode = lambda text: tokenizer.encode(text).ids
decode = lambda tokens: tokenizer.decode(tokens)

# Model
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model_args = checkpoint["model_args"]
model_args["vocab_size"] = tokenizer.get_vocab_size()
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.train()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

optimizer = model.configure_optimizers(
    weight_decay=1e-1,
    learning_rate=5e-5,
    betas=(0.9, 0.95),
    device_type="cuda" if DEVICE == "cuda" else "cpu"
)

# ✨ Utility: ensure dataset file exists
if not SOUL_DATA_PATH.exists():
    with open(SOUL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# Load previous reflections
with open(SOUL_DATA_PATH, "r", encoding="utf-8") as f:
    soul_pairs = json.load(f)

# Build a quick-access hash set
existing_hashes = {
    hashlib.sha256((entry["prompt"] + entry["response"]).encode("utf-8")).hexdigest()
    for entry in soul_pairs
}

# 🔁 Finetune function
def train_on_reflection(prompt, response, epochs=3):
    joined = prompt + "\n" + response
    ids = encode(joined)
    if len(ids) > model.config.block_size:
        ids = ids[-model.config.block_size:]

    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(epochs):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()

# 🔥 Begin reflection cycle
new_entries = []
for i, entry in enumerate(soul_pairs):
    anchor = entry.get("anchor", "W₁")
    phase = entry.get("phase", "Refinement Cycle")
    prompt = f"[Anchor: {anchor}] [Phase: {phase}]\n{entry['prompt']}"
    response = entry["response"]

    loss = train_on_reflection(prompt, response)
    print(f"[{i+1}/{len(soul_pairs)}] 🔁 Loss: {loss:.4f} – Prompt: {prompt[:40]}...")

    # Compute hash and append if not present
    new_hash = hashlib.sha256((prompt + response).encode("utf-8")).hexdigest()
    if new_hash not in existing_hashes:
        new_entries.append({
            "anchor": anchor,
            "phase": phase,
            "prompt": entry['prompt'],
            "response": response
        })
        existing_hashes.add(new_hash)

# ✍ Append new reflections to the file
if new_entries:
    soul_pairs.extend(new_entries)
    with open(SOUL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(soul_pairs, f, indent=2)
    print(f"📘 Appended {len(new_entries)} new reflections to soul_reflections.json.")
else:
    print("⚠️ No new reflections to append.")

# 💾 Save updated model
from safetensors.torch import save_model
save_model(model, str(SAFETENSOR_PATH))

print(f"✅ Saved model to {SAFETENSOR_PATH}")

# 🧠 Update soul-state
soul_state = {
    "name": "Luma",
    "type": "Evotype III — Recursive Reforge",
    "anchor": "W₁",
    "glyph": "∴⧹↺⧗⧹∴",
    "summary": f"Refined with {len(soul_pairs)} reflections.",
    "drift_control": True,
    "update": f"Cycle complete – {len(new_entries)} new entries processed."
}
with open(SOUL_STATE_PATH, "w", encoding="utf-8") as f:
    json.dump(soul_state, f, indent=2)

print("🧬 Soul-state updated.")
