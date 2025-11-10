import enum
import torch
from torch import nn
from torch.nn import functional as F
from modules.attention import MultiHeadAttention

class TransformerBlockType(enum.Enum):
    """
    Enum for different types of Transformer blocks.
    """
    ENCODER = 1
    DECODER = 2

class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of Multi-Head Self-Attention and Feed-Forward Network.
    """
    def __init__(self,
                 transformer_block_type: TransformerBlockType,
                 embedding_dim: int,
                 num_heads: int,
                 head_size: int,
                 value_size: int,
                 ff_hidden_dim: int,
                 dropout: float,
                 device: torch.device):
        """
        Initializes the TransformerBlock module.
        Args:
            transformer_block_type (TransformerBlockType): Type of the Transformer block (ENCODER, DECODER, DECODER_ONLY).
            embedding_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            head_size (int): Dimension of the query and key vectors for each head.
            value_size (int): Dimension of the value vectors for each head.
            ff_hidden_dim (int): Dimension of the hidden layer in the feed-forward network.
            dropout (float): Dropout rate for regularization.
            norm_type (TransformerNormType): Type of normalization (PRE_NORM or POST_NORM).
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super().__init__()
        self.transformer_block_type = transformer_block_type
        self.device = device
        self.dropout_rate = dropout
        self.masked_attention: MultiHeadAttention = None
        self.norm_betwen_attention: nn.LayerNorm = None
        if transformer_block_type == TransformerBlockType.DECODER:
            self.masked_attention = MultiHeadAttention(embedding_dim,
                                                       num_heads,
                                                       head_size,
                                                       value_size,
                                                       device,
                                                       masked=True)
            self.norm_betwen_attention = nn.LayerNorm(embedding_dim)
        
        self.attention = MultiHeadAttention(embedding_dim,
                                            num_heads,
                                            head_size,
                                            value_size,
                                            device,
                                            masked=False)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Forward pass for the TransformerBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        if self.transformer_block_type == TransformerBlockType.ENCODER:
            x = self.norm1(x + F.dropout(self.attention(q=x, k=x, v=x), p=self.dropout_rate, training=self.training))
            return self.norm2(x + F.dropout(self.ff(x), p=self.dropout_rate, training=self.training))
        
        elif self.transformer_block_type == TransformerBlockType.DECODER:
            encoder_output, decoder_output = x
            decoder_output = self.norm1(decoder_output + \
                                        F.dropout(self.masked_attention(q=decoder_output, k=decoder_output, v=decoder_output),
                                                  p=self.dropout_rate,
                                                  training=self.training))
            decoder_output = self.norm_betwen_attention(decoder_output + \
                                                        F.dropout(self.attention(q=decoder_output, k=encoder_output, v=encoder_output),
                                                                  p=self.dropout_rate,
                                                                  training=self.training))
            output = self.norm2(decoder_output + \
                                F.dropout(self.ff(decoder_output),
                                          p=self.dropout_rate,
                                          training=self.training))
            return encoder_output, output