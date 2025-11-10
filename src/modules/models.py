from typing import Type, Union, Callable, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from encoders.text import TextEncoder
from modules.transformer import TransformerBlock, TransformerBlockType
from tokenizers.base import Tokenizer

class TranslationTransformer(nn.Module):
    """
    Implementation of a Transformer model for translation tasks.
    """
    def __init__(self,
                 sequence_length: int,
                 vocab_size: int,
                 text_encoder: Type[TextEncoder],
                 embedding_dim: int,
                 n_encoder_blocks: int,
                 n_decoder_blocks: int,
                 num_encoder_heads: int,
                 num_decoder_heads: int,
                 encoder_head_size: int,
                 decoder_head_size: int,
                 encoder_value_size: int,
                 decoder_value_size: int,
                 encoder_ff_hidden_dim: int,
                 decoder_ff_hidden_dim: int,
                 dropout_rate: float,
                 device: torch.device):
        """
        Initializes the LanguageModel.
        Args:
            sequence_length (int): Length of input sequences.
            vocab_size (int): Size of the vocabulary.
            text_encoder (Type[TextEncoder]): TextEncoder class to convert tokens to embeddings.
            embedding_dim (int): Dimension of the token embeddings.
            n_encoder_blocks (int): Number of Transformer encoder blocks.
            n_decoder_blocks (int): Number of Transformer decoder blocks.
            num_encoder_heads (int): Number of attention heads in the encoder blocks.
            num_decoder_heads (int): Number of attention heads in the decoder blocks.
            encoder_head_size (int): Size of each attention head in the encoder blocks.
            decoder_head_size (int): Size of each attention head in the decoder blocks.
            encoder_value_size (int): Size of the value vectors in the encoder blocks.
            decoder_value_size (int): Size of the value vectors in the decoder blocks.
            encoder_ff_hidden_dim (int): Hidden dimension of the feed-forward networks in the encoder blocks.
            decoder_ff_hidden_dim (int): Hidden dimension of the feed-forward networks in the decoder blocks
            dropout_rate (float): Dropout rate to use in the Transformer blocks.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super().__init__()
        assert dropout_rate >= 0.0 and dropout_rate < 1.0, "Dropout rate must be in [0.0, 1.0)"
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.device = device
        self.text_encoder = text_encoder(vocab_size=self.vocab_size,
                                         embedding_dim=embedding_dim)
        self.positional_encoder = nn.Embedding(sequence_length,
                                               embedding_dim)
        self.encoder_transformer_blocks = nn.Sequential(*[
            TransformerBlock(transformer_block_type=TransformerBlockType.ENCODER,
                             embedding_dim=embedding_dim,
                             num_heads=num_encoder_heads,
                             head_size=encoder_head_size,
                             value_size=encoder_value_size,
                             ff_hidden_dim=encoder_ff_hidden_dim,
                             dropout=dropout_rate,
                             device=self.device)
            for _ in range(n_encoder_blocks)
        ])
        self.decoder_transformer_blocks = nn.Sequential(*[
            TransformerBlock(transformer_block_type=TransformerBlockType.DECODER,
                             embedding_dim=embedding_dim,
                             num_heads=num_decoder_heads,
                             head_size=decoder_head_size,
                             value_size=decoder_value_size,
                             ff_hidden_dim=decoder_ff_hidden_dim,
                             dropout=dropout_rate,
                             device=self.device)
            for _ in range(n_decoder_blocks)
        ])
        self.output_layer = nn.Linear(embedding_dim, self.vocab_size)


    def forward(self,
                encoder_input_sequence: torch.Tensor,
                decoder_input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TranslationTransformer.
        Args:
            encoder_input_sequence (torch.Tensor): Input tensor for the encoder of shape (B, T).
            decoder_input_sequence (torch.Tensor): Input tensor for the decoder of shape (B, T).
        Returns:
            torch.Tensor: Output logits of shape (B, T, vocab_size).
        """
        positions = self.positional_encoder(torch.arange(self.sequence_length,
                                                         device=self.device)) # (B, T, embedding_dim)
        encoder_embeddings = self.text_encoder(encoder_input_sequence)  # (B, T, embedding_dim)
        encoder_embeddings = encoder_embeddings + positions  # (B, T, embedding_dim)
        encoder_output = self.encoder_transformer_blocks(encoder_embeddings)  # (B, T, embedding_dim)

        decoder_embeddings = self.text_encoder(decoder_input_sequence)  # (B, T, embedding_dim)
        decoder_embeddings = decoder_embeddings + positions  # (B, T, embedding_dim)
        decoder_embeddings = self.decoder_transformer_blocks([encoder_output,
                                                             decoder_embeddings])  # (B, T, embedding_dim)
        # Discard the encoder output that is returned alongside the decoder embeddings
        _, decoder_embeddings = decoder_embeddings
        logits = self.output_layer(decoder_embeddings)  # (B, T, vocab_size)
        return logits
    
    @staticmethod
    def create_from_yaml(yaml_path: str,
                         text_encoder: Type[TextEncoder],
                         vocab_size: int,
                         device: torch.device) -> "TranslationTransformer":
        """
        Creates a TranslationTransformer instance from a YAML configuration file.
        Args:
            yaml_path (str): Path to the YAML configuration file.
            text_encoder (Type[TextEncoder]): TextEncoder class to convert tokens to embeddings.
            vocab_size (int): Size of the vocabulary.
            device (torch.device): Device to run the model on (CPU or GPU).
        Returns:
            TranslationTransformer: An instance of the TranslationTransformer class.
        """
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        return TranslationTransformer(
            sequence_length=config['sequence_length'],
            vocab_size=vocab_size,
            text_encoder=text_encoder,
            embedding_dim=config['embedding_dim'],
            n_encoder_blocks=config['encoder']['n_blocks'],
            n_decoder_blocks=config['decoder']['n_blocks'],
            num_encoder_heads=config['encoder']['num_heads'],
            num_decoder_heads=config['decoder']['num_heads'],
            encoder_head_size=config['encoder']['head_size'],
            decoder_head_size=config['decoder']['head_size'],
            encoder_value_size=config['encoder']['value_size'],
            decoder_value_size=config['decoder']['value_size'],
            encoder_ff_hidden_dim=config['encoder']['ff_hidden_dim'],
            decoder_ff_hidden_dim=config['decoder']['ff_hidden_dim'],
            dropout_rate=config['dropout_rate'],
            device=device
        )
    
    def fit(self,
            dataset: Union[torch.utils.data.DataLoader, Callable[[str], Tuple[torch.Tensor, torch.Tensor]]],
            optimizer: torch.optim.Optimizer,
            iterations: int,
            evaluation_interval: int,
            evaluation_iterations: int,
            best_model_path: str = "best_model.pth",
            log_losses_on_file: bool = False) -> Tuple[List[float], List[float]]:
        """
        Trains the TranslationTransformer model.
        Args:
            dataset (Union[torch.utils.data.DataLoader, Callable[[str], Tuple[torch.Tensor, torch.Tensor]]]):
                Dataset or callable that returns batches of (input, target) tensors.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            iterations (int): Number of training iterations.
            evaluation_interval (int): Interval (in iterations) at which to evaluate on the validation set.
            evaluation_iterations (int): Number of iterations to use for validation evaluation.
            best_model_path (str): Path to save the best model based on validation loss.
            log_losses_on_file (bool): Whether to log losses on a text file or not.
        Returns:
            Tuple[List[float], List[float]]: Training loss history and validation loss history.
        """
        self.to(self.device)
        self.train()
        loss = torch.nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        training_loss_history = []
        val_loss_history = []
        training_loss = 0.0

        for iteration in (training_pbar := tqdm(range(iterations), desc="Iterations")):
            x, y, y_t = dataset.get_batch("train")
            B, T = x.shape
            x = x.to(self.device)
            target = y_t.contiguous().view(B * T).to(self.device)
            logits = self(x, y.to(self.device)).contiguous().view(B * T, self.vocab_size)
            loss_value = loss(logits, target)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            training_loss += loss_value.item()

            if iteration % evaluation_interval == 0 and iteration > 0:
                # Log training loss
                training_loss /= evaluation_interval
                training_loss_history.append(training_loss)
                
                # Evaluate on validation set
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(evaluation_iterations):
                        x, y, y_t = dataset.get_batch("val")
                        B, T = x.shape
                        x = x.to(self.device)
                        target = y_t.contiguous().view(B * T).to(self.device)
                        logits = self(x, y.to(self.device)).contiguous().view(B * T, self.vocab_size)
                        loss_value = loss(logits, target)
                        val_loss += loss_value.item()

                # Print training and validation loss in the progress bar and log
                # validation loss to a file.
                val_loss /= evaluation_iterations
                val_loss_history.append(val_loss)

                training_pbar.set_postfix({"Loss": training_loss, "Val Loss": val_loss})

                if log_losses_on_file:
                    with open("training_losses.txt", "a") as f:
                        f.write(f"{iteration}:{training_loss}\n")

                    with open("val_losses.txt", "a") as f:
                        f.write(f"{iteration}:{val_loss}\n")

                training_loss = 0.0

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), best_model_path)
                self.train()

        return training_loss_history, val_loss_history
    
    def translate(self,
                  input_sequence: torch.Tensor,
                  tokenizer: Tokenizer) -> torch.Tensor:
        self.to(self.device)
        self.eval()
        decoder_input_sequence = torch.tensor([tokenizer.get_start_token()],
                                              dtype=torch.long).unsqueeze(0).to(self.device)  # (1, 1)
        encoder_input_sequence = input_sequence.unsqueeze(0).to(self.device)  # (1, T)
        
        while True:
            with torch.no_grad():
                if decoder_input_sequence.shape[1] < self.sequence_length:
                    input_dec_tokens_crop_or_pad = tokenizer.pad_sequence(decoder_input_sequence.squeeze(0),
                                                                          desired_length=self.sequence_length)  # (1, sequence_length)
                    next_token_index = decoder_input_sequence.shape[1] - 1
                elif decoder_input_sequence.shape[1] > self.sequence_length:
                    input_dec_tokens_crop_or_pad = decoder_input_sequence[:, -self.sequence_length:]  # (1, sequence_length)
                    next_token_index = self.sequence_length - 1

                logits = self(encoder_input_sequence, input_dec_tokens_crop_or_pad)  # (1, sequence_length, vocab_size)
                next_token_logits = logits[0, next_token_index, :]  # (vocab_size,)
                next_token_distrubution = F.softmax(next_token_logits, dim=0)  # (vocab_size,)
                next_token_id = torch.multinomial(next_token_distrubution, num_samples=1).unsqueeze(0)  # (1, 1)
                decoder_input_sequence = torch.cat((decoder_input_sequence, next_token_id), dim=1)  # (1, sequence_length + 1)

                if next_token_id.item() == tokenizer.get_end_token():
                    return decoder_input_sequence.squeeze(0)  # (T,)
