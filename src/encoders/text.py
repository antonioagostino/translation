from abc import ABC, abstractmethod
import torch
from torch import nn

class TextEncoder(ABC, nn.Module):
    """
    Base class for text encoders.
    This class provides a basic interface for encoding text data.
    It should be extended by specific encoder implementations.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input text into embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of token indices.
        
        Returns:
            torch.Tensor: Output tensor of embeddings.
        """
        pass

class SimpleTextEncoder(TextEncoder):
    """
    A simple text encoder that converts token indices to embeddings.
    This encoder uses an embedding layer to transform token indices into dense vectors.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int):
        """
        Initializes the SimpleTextEncoder with a vocabulary size and embedding dimension.   
        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of the embeddings.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input text into embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of token indices.
        
        Returns:
            torch.Tensor: Output tensor of embeddings.
        """
        assert x.dtype == torch.long, "Input tensor must contain long integers (token indices)"

        # Shape: bsize x seq_len x embedding_dim
        return self.token_embedding(x)