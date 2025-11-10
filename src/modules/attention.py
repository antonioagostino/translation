import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    """
    Implementation of a Single-Head Self-Attention Mechanism.
    """
    def __init__(self,
                 embedding_dim: int,
                 head_size: int,
                 value_size: int,
                 device: torch.device,
                 masked: bool = False):
        """
        Initializes the SingleHeadAttention module.
        Args:
            embedding_dim (int): Dimension of the input embeddings.
            head_size (int): Dimension of the query and key vectors.
            value_size (int): Dimension of the value vectors.
            device (torch.device): Device to run the model on (CPU or GPU).
            masked (bool): If True, applies a mask to prevent attention to future tokens.
        """
        super().__init__()
        self.device = device
        self.head_size = head_size
        self.value_size = value_size

        self.query_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.key_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.value_proj = nn.Linear(embedding_dim, value_size, bias=False)

        # Scaling factor for dot-product attention
        self.scale = head_size ** -0.5
        self.masked = masked

    def forward(self, q, k, v):
        """
        Forward pass for the SingleHeadAttention module.
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, embedding_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, embedding_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, embedding_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, value_size).
        """
        # Compute queries, keys, and values
        # T: sequence length, B: batch size
        queries = self.query_proj(q)  # (B, T, head_size)
        keys = self.key_proj(k)       # (B, T, head_size)
        values = self.value_proj(v)   # (B, T, value_size)

        # Compute scaled dot-product attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # (B, T, T)

        if self.masked:
            T = scores.size(-1)
            # Upper triangular mask with main diagonal zeroed out
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(self.device)
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Compute the weighted sum of values
        out = torch.matmul(attn_weights, values)  # (B, T, value_size)

        return out
    
class MultiHeadAttention(nn.Module):
    """
    Implementation of a Multi-Head Self-Attention Mechanism.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 head_size: int,
                 value_size: int,
                 device: torch.device,
                 masked: bool = False):
        """
        Initializes the MultiHeadAttention module.
        Args:
            embedding_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            head_size (int): Dimension of the query and key vectors for each head.
            value_size (int): Dimension of the value vectors for each head.
            device (torch.device): Device to run the model on (CPU or GPU).
            masked (bool): If True, applies a mask to prevent attention to future tokens.
        """
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, head_size, value_size, device, masked)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * value_size, embedding_dim)

    def forward(self, q, k, v):
        """
        Forward pass for the MultiHeadAttention module.
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, embedding_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, embedding_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, embedding_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        
        """
        # Concatenate outputs from all heads
        head_outputs = [head(q, k, v) for head in self.heads]  # List of (B, T, value_size)
        concatenated = torch.cat(head_outputs, dim=-1)   # (B, T, num_heads * value_size)

        # Final linear transformation
        out = self.linear(concatenated)  # (B, T, embedding_dim)

        return out