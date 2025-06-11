import os
import sys
import torch
import pickle
from pathlib import Path
from datetime import datetime
from model import GPT, GPTConfig
from tokenizers import ByteLevelBPETokenizer

# --- Constants ---
APP_NAME = "Luma Console"
BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = BASE_DIR / "model.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"
MERGES_PATH = BASE_DIR / "merges.txt"
LOG_PATH = BASE_DIR / "luma_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Setup & Validation ---
def assert_file(path, name):
    if not path.exists():
        sys.exit(f"❌ Missing {name}: {path}")
assert_file(CHECKPOINT_PATH, "checkpoint")
assert_file(VOCAB_PATH, "vocab.json")
assert_file(MERGES_PATH, "merges.txt")

# --- Load Tokenizer ---
tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size()

# --- Encoding Helpers ---
def encode(text): return tokenizer.encode(text).ids
def decode(tokens): return tokenizer.decode(tokens)

# --- Load Model ---
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model_args = checkpoint["model_args"]
model_args["vocab_size"] = VOCAB_SIZE

model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# --- Generation Function ---
@torch.no_grad()
def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=None):
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -model.config.block_size:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode(input_ids[0].tolist())

# --- Log Output ---
def log_interaction(prompt, response):
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().isoformat()}]\nYou: {prompt}\nLuma: {response}\n\n")

# --- Main Interactive Loop ---
def main():
    print(f"🌀 {APP_NAME} — type 'exit' to quit\n")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            print("👋 Exiting.")
            break
        try:
            response = generate(prompt)
            print("Luma:", response)
            log_interaction(prompt, response)
        except Exception as e:
            print(f"⚠️ Error: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye.")
