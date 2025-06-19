from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json

app = FastAPI(title="Poetry Generator", description="AI Poetry Generator using GPT model")

# Load model configuration and vocabulary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model hyperparameters (same as in model.py)
block_size = 512
n_embd = 756
n_head = 12
n_layer = 12
dropout = 0.1

# Load vocabulary
def load_vocabulary():
    """Load vocabulary from the original dataset"""
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            text = f.read().lower()  # Convert to lowercase for consistency
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return vocab_size, stoi, itos
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset file not found. Please ensure dataset.txt exists.")

# Initialize vocabulary
vocab_size, stoi, itos = load_vocabulary()
encode = lambda s: [stoi[c] for c in s.lower()]  # Convert to lowercase before encoding
decode = lambda l: ''.join([itos[i] for i in l])

# Model classes (copied from model.py)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = GPTLanguageModel()
model = model.to(device)

# Load trained model if available
def load_trained_model():
    model_paths = ['final_model.pt', 'model_checkpoint.pt', 'best_model.pt']
    for path in model_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model from {path}")
                return True
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    print("No trained model found. Using randomly initialized model.")
    return False

# Load model on startup
load_trained_model()
model.eval()

# Request/Response models
class PoetryRequest(BaseModel):
    prompt: str
    max_length: int = 300
    temperature: float = 1.0

class PoetryResponse(BaseModel):
    generated_text: str
    original_prompt: str
    vocab_size: int

@app.get("/")
async def root():
    return {
        "message": "Welcome to Poetry Generator API",
        "endpoints": {
            "/api": "GET - API information",
            "/health": "GET - Health check",
            "/generate": "POST - Generate poetry",
            "/generate_creative": "POST - Generate creative poetry"
        }
    }

@app.get("/api")
async def api_info():
    return {
        "message": "Poetry Generator API", 
        "vocab_size": vocab_size,
        "device": device,
        "endpoints": {
            "/generate": "POST - Generate poetry from text prompt",
            "/health": "GET - Check API health",
            "/generate_creative": "POST - Generate creative poetry with temperature control"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vocab_size": vocab_size,
        "device": device,
        "model_loaded": True
    }

@app.post("/generate", response_model=PoetryResponse)
async def generate_poetry(request: PoetryRequest):
    try:
        # Preprocess input: convert to lowercase and limit length
        prompt = request.prompt.lower().strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Limit prompt length to avoid memory issues
        if len(prompt) > block_size - 50:
            prompt = prompt[:block_size - 50]
        
        # Encode the prompt
        try:
            encoded_prompt = encode(prompt)
        except KeyError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Prompt contains unsupported characters. Please use only characters from the training vocabulary."
            )
        
        # Convert to tensor
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=request.max_length)
            generated_text = decode(generated[0].tolist())
        
        return PoetryResponse(
            generated_text=generated_text,
            original_prompt=request.prompt,
            vocab_size=vocab_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Add temperature control for more creative generation
@app.post("/generate_creative", response_model=PoetryResponse)
async def generate_creative_poetry(request: PoetryRequest):
    try:
        prompt = request.prompt.lower().strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if len(prompt) > block_size - 50:
            prompt = prompt[:block_size - 50]
        
        try:
            encoded_prompt = encode(prompt)
        except KeyError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Prompt contains unsupported characters."
            )
        
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        # Generate with temperature control
        with torch.no_grad():
            generated_tokens = []
            current_context = context
            
            for _ in range(request.max_length):
                idx_cond = current_context[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / request.temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                current_context = torch.cat((current_context, idx_next), dim=1)
                generated_tokens.append(idx_next.item())
            
            full_generated = context[0].tolist() + generated_tokens
            generated_text = decode(full_generated)
        
        return PoetryResponse(
            generated_text=generated_text,
            original_prompt=request.prompt,
            vocab_size=vocab_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Creative generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for Hugging Face Spaces
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port) 