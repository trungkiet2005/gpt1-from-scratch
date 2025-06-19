from model import GPTLanguageModel
import torch
import os
from model import encode, decode, learning_rate, device


# Load the saved model
final_model_path = 'final_model.pt'
checkpoint = torch.load(final_model_path, map_location=torch.device('cpu'))

# Initialize model and load state dict
model = GPTLanguageModel()
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Initialize optimizer and load state dict
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Model loaded from {final_model_path}")


context = torch.tensor([encode("mùa thu mẹ ru ")], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))