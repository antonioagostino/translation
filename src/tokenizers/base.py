from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from pathlib import Path
import json
import torch

class Tokenizer(ABC):
    """
    Base class for tokenizers.
    This class provides a basic interface for tokenization and detokenization.
    It should be extended by specific tokenizer implementations.
    """
    def __init__(self,
                 vocabulary_map: Optional[Union[str, Dict[int, str]]] = None):
        """
        Initializes the tokenizer loading vocabulary from a file if provided.
        Args:
            vocabulary_filepath (Optional[str]): Path to the vocabulary file.
                If None, the vocabulary will be created with a training dataset.
        """
        self.stoi = {}  # String to int
        self.itos = {}  # Int to string
        self.vocab_size = 0
        self.vocabulary = []
        
        if not vocabulary_map is None:
            if isinstance(vocabulary_map, dict):
                self.itos = vocabulary_map
            else:
                assert isinstance(vocabulary_map, str), "vocabulary_filepath should be a string or a dictionary."
                assert Path(vocabulary_map).is_file(), f"Vocabulary file '{vocabulary_map}' does not exist."
                itos_mapping = {}
                with open(vocabulary_map, 'r', encoding='utf-8') as f:
                    itos_mapping = json.load(f)
                    
                for i, ch in itos_mapping.items():
                    self.itos[int(i)] = ch
                    self.stoi[ch] = int(i)
                
                self.vocabulary = self.itos.values()
                self.vocab_size = len(self.vocabulary)

    @abstractmethod
    def tokenize(self,
                 text: str) -> torch.Tensor:
        """
        Tokenizes a string into a tensor of tokens.
        Args:
            text (str): The input string to tokenize.
        Returns:
            torch.Tensor: A tensor of tokens represented as integers.
        """
        pass

    @abstractmethod
    def detokenize(self, tokens: torch.Tensor) -> str:
        """
        Converts a tensor of tokens back into a string.
        Args:
            tokens (torch.Tensor): A tensor of tokens represented as integers.
        Returns:
            str: The detokenized string.
        """
        pass

    @staticmethod
    @abstractmethod
    def train_tokenizer(corpus: str) -> "Tokenizer":
        """
        Trains the tokenizer and builds the vocabulary from a given text corpus.
        Args:
            corpus (str): The text corpus to build the vocabulary from.
        """
        pass

    @abstractmethod
    def add_delimiters(self,
                       text: str) -> str:
        """
        Adds start and end tokens to the text.
        Args:
            text (str): The input string.
        Returns:
            str: The string with start and end tokens added.
        """
        pass

    @abstractmethod
    def pad_sequence(self,
                     tokens: torch.Tensor,
                     desired_length: int) -> torch.Tensor:
        """
        Pads the token sequence to the desired length with the pad token.
        Args:
            tokens (torch.Tensor): The tensor of tokens to pad.
            desired_length (int): The desired length of the token sequence.
        Returns:
            torch.Tensor: The padded tensor of tokens.
        """
        pass

    def save_vocabulary_map(self, filepath: str) -> None:
        """
        Saves the vocabulary to a file in JSON format.
        Args:
            filepath (str): The path to the vocabulary map file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.itos, f, indent=4)

    @abstractmethod
    def get_start_token(self) -> int:
        """
        Returns the special start token.
        Returns:
            int: The start token.
        """
        pass
    
    @abstractmethod
    def get_end_token(self) -> int:
        """
        Returns the special end token.
        Returns:
            int: The end token.
        """
        pass

class CharacterTokenizer(Tokenizer):
    """
    A simple character-level tokenizer.
    This tokenizer treats each character as a token.
    """
    def __init__(self,
                 vocabulary_map: Optional[Union[str, Dict[int, str]]] = None):
        """
        Initializes the tokenizer loading vocabulary from a file if provided.
        Args:
            vocabulary_filepath (Optional[str]): Path to the vocabulary file.
                If None, the vocabulary will be created with a training dataset.
        """
        super().__init__(vocabulary_map)
        self.special_tokens = ["<PAD>", "<START>", "<END>"]

    
    def tokenize(self,
                 text: str) -> torch.Tensor:
        """
        Tokenizes a string into a tensor of tokens.
        Args:
            text (str): The input string to tokenize.
        Returns:
            torch.Tensor: A tensor of tokens represented as integers.
        """
        tokens = []
        i = 0
        while i < len(text):
            special_token_found = False
            if text[i] == "<":
                for special_token in self.special_tokens:
                    if text[i:i+len(special_token)] == special_token:
                        tokens.append(self.stoi[special_token])
                        i += len(special_token)
                        special_token_found = True
                        break

            if not special_token_found:
                tokens.append(self.stoi[text[i]])
                i += 1
        
        return torch.tensor(tokens, dtype=torch.long)

    def detokenize(self, tokens: torch.Tensor) -> str:
        """
        Converts a tensor of tokens back into a string.
        Args:
            tokens (torch.Tensor): A tensor of tokens represented as integers.
        Returns:
            str: The detokenized string.
        """
        assert isinstance(tokens, torch.Tensor), "Input tokens should be a torch.Tensor"
        assert tokens.dtype == torch.long, "Input tokens tensor should be of type torch.long"
        text = ""
        tokens = tokens.tolist()
        for token in tokens:
            text += self.itos[token]
        return text

    @staticmethod
    def train_tokenizer(corpus: str) -> "Tokenizer":
        """
        Trains the tokenizer and builds the vocabulary from a given text corpus.
        Args:
            corpus (str): The text corpus to build the vocabulary from.
        """
        unique_chars = sorted(list(set(corpus)))
        # Add special tokens
        vocabulary = ["<PAD>", "<START>", "<END>"] + unique_chars
        itos = {i:ch for i, ch in enumerate(vocabulary)}
        return CharacterTokenizer(vocabulary_map=itos)

    def add_delimiters(self,
                       text: str) -> str:
        """
        Adds start and end tokens to the text.
        Args:
            text (str): The input string.
        Returns:
            str: The string with start and end tokens added.
        """
        return f"<START>{text}<END>"
    
    def pad_sequence(self,
                     tokens: torch.Tensor,
                     desired_length: int) -> torch.Tensor:
        """
        Pads the token sequence to the desired length with the pad token.
        Args:
            tokens (torch.Tensor): The tensor of tokens to pad.
            desired_length (int): The desired length of the token sequence.
        Returns:
            torch.Tensor: The padded tensor of tokens.
        """
        assert isinstance(tokens, torch.Tensor), "Input tokens should be a torch.Tensor"
        assert tokens.ndim == 1, "Input tokens tensor should be 1-dimensional"
        assert tokens.shape[0] <= desired_length, "The length of tokens exceeds the desired length."
        pad_token_id = self.stoi["<PAD>"]

        padding = torch.full((desired_length - tokens.shape[0], ), pad_token_id, dtype=torch.long)
        new_sequence = torch.cat((tokens, padding))
        return new_sequence
    
    def get_start_token(self) -> int:
        """
        Returns the special start token.
        Returns:
            int: The start token.
        """
        return self.stoi["<START>"]
    
    def get_end_token(self) -> int:
        """
        Returns the special end token.
        Returns:
            int: The end token.
        """
        return self.stoi["<END>"]