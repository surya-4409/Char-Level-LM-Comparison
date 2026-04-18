import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        # tril is a lower triangular matrix used for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))
        
        # Apply causal mask: don't look into the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # Aggregate the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embed_size)

    def forward(self, x):
        # Concatenate the outputs of all heads over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size), # Projection layer back to embed_size
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (MHA) followed by computation (FFWD) """
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, embed_size, block_size)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Residual connections around the sub-layers
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size):
        """
        Initializes the Mini-Transformer model.
        
        Args:
            vocab_size (int): Number of unique characters.
            embed_size (int): Dimension of character embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer blocks.
            block_size (int): Maximum sequence length (context window).
        """
        super().__init__()
        self.block_size = block_size
        
        # Token embedding and Positional embedding
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        
        # Sequential blocks
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads, block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size) # Final layer norm
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, hidden=None):
        B, T = x.shape
        
        # x and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(x) # (B, T, C)
        
        # Generate positional integers [0, 1, 2, ..., T-1] and embed them
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb # (B, T, C)
        
        # Pass through Transformer blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)   # (B, T, C)
        
        # Final linear projection
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        return logits, None # Returning None for hidden state to keep API identical to LSTM