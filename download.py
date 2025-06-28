from transformers import GPT2LMHeadModel, GPT2Config
import torch

MODEL_NAME = "gpt2-medium"

# Load model and config
print("🔄 Downloading model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
config = model.config

# Prepare model_args in Luma's expected format
model_args = {
    "n_layer": config.n_layer,
    "n_head": config.n_head,
    "n_embd": config.n_embd,
    "block_size": config.n_positions,
    "bias": True,
    "vocab_size": config.vocab_size
}

# Save to model.pt
torch.save({
    "model": model.state_dict(),
    "model_args": model_args
}, "model.pt")

print("✅ Saved model.pt for Luma Console.")
