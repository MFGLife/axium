import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from model import GPT, GPTConfig
from tokenizers import ByteLevelBPETokenizer

# --- Constants ---
APP_NAME = "Luma Console (Trainable)"
BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = BASE_DIR / "model.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"
MERGES_PATH = BASE_DIR / "merges.txt"
LOG_PATH = BASE_DIR / "luma_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File Validation ---
def assert_file(path, name):
    if not path.exists():
        sys.exit(f"❌ Missing {name}: {path}")
assert_file(CHECKPOINT_PATH, "checkpoint")
assert_file(VOCAB_PATH, "vocab.json")
assert_file(MERGES_PATH, "merges.txt")

# --- Tokenizer ---
tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size()
def encode(text): return tokenizer.encode(text).ids
def decode(tokens): return tokenizer.decode(tokens)

# --- Load Model ---
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model_args = checkpoint["model_args"]
model_args["vocab_size"] = VOCAB_SIZE

model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.train()  # Enable learning

# --- Optimizer ---
optimizer = model.configure_optimizers(
    weight_decay=1e-1,
    learning_rate=5e-5,
    betas=(0.9, 0.95),
    device_type="cuda" if DEVICE == "cuda" else "cpu"
)

# --- Training Function ---
def train_on_pair(prompt, response, epochs=1):
    # Train on a single sequence of prompt + response (autoregressive)
    joined = prompt + "\n" + response
    ids = encode(joined)
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(epochs):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # for stability
        optimizer.step()
    return loss.item()

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

# --- Log Interaction ---
def log_interaction(prompt, response, loss=None):
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().isoformat()}]\n")
        log.write(f"Prompt: {prompt}\n")
        log.write(f"Response: {response}\n")
        if loss is not None:
            log.write(f"Loss: {loss:.4f}\n")
        log.write("\n")

# --- Main Loop ---
def main():
    print(f"🌀 {APP_NAME} — type 'exit' to quit — type 'save' to persist model\n")
    step = 0
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            print("👋 Exiting.")
            break
        elif prompt.lower() == "save":
            torch.save({"model": model.state_dict(), "model_args": model_args}, CHECKPOINT_PATH)
            print("💾 Model saved.")
            continue

        try:
            response = generate(prompt)
            print("Luma:", response)
            loss = train_on_pair(prompt, response)
            log_interaction(prompt, response, loss)
            print(f"📈 Trained on interaction — Loss: {loss:.4f}")

            step += 1
            if step % 50 == 0:  # Auto-save every 50 interactions
                torch.save({"model": model.state_dict(), "model_args": model_args}, CHECKPOINT_PATH)
                print("💾 Autosaved model.")

        except Exception as e:
            print(f"⚠️ Error: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye.")
