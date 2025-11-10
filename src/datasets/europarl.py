from typing import Tuple
from random import randint
import torch
import pandas as pd
from tokenizers.base import Tokenizer

class EuroParlTranslationDataset:
    """
    A class for handling the EuroParl translation dataset in CSV format.
    """
    def __init__(self,
                 train_data_path: str,
                 val_data_path: str,
                 test_data_path: str,
                 sequence_length: int,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 already_tokenized: bool):
        """
        Initializes the EuroParlTranslationDataset.
        Args:
            train_data_path (str): Path to the CSV file containing training data.
            val_data_path (str): Path to the CSV file containing validation data.
            test_data_path (str): Path to the CSV file containing test data.
            sequence_length (int): Length of each sequence to be created from the text.
            batch_size (int): Batch size for data loading.
            tokenizer (Tokenizer): A tokenizer class to convert characters to tokens.
            already_tokenized (bool): Whether the dataset was already tokenized or not.
        """
        super().__init__()
        train_dataframe = pd.read_csv(train_data_path)
        train_italian_sentences = train_dataframe["sent_it"].to_list()
        train_english_sentences = train_dataframe["sent_en"].to_list()
        assert len(train_italian_sentences) == len(train_english_sentences), "The number of Italian and English sentences must be the same."

        val_dataframe = pd.read_csv(val_data_path)
        val_italian_sentences = val_dataframe["sent_it"].to_list()
        val_english_sentences = val_dataframe["sent_en"].to_list()
        assert len(val_italian_sentences) == len(val_english_sentences), "The number of Italian and English sentences must be the same."

        test_dataframe = pd.read_csv(test_data_path)
        test_italian_sentences = test_dataframe["sent_it"].to_list()
        test_english_sentences = test_dataframe["sent_en"].to_list()
        assert len(test_italian_sentences) == len(test_english_sentences), "The number of Italian and English sentences must be the same."

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        if not already_tokenized:
            self.train_ita_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in train_italian_sentences]
            self.train_en_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in train_english_sentences]
            self.val_ita_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in val_italian_sentences]
            self.val_en_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in val_english_sentences]
            self.test_ita_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in test_italian_sentences]
            self.test_en_data = [self.tokenizer.tokenize(self.tokenizer.add_delimiters(sequence)) for sequence in test_english_sentences]
        else:
            self.train_ita_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in train_italian_sentences]
            self.train_en_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in train_english_sentences]
            self.val_ita_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in val_italian_sentences]
            self.val_en_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in val_english_sentences]
            self.test_ita_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in test_italian_sentences]
            self.test_en_data = [torch.tensor(list(map(int, sequence.split())), dtype=torch.long) for sequence in test_english_sentences]

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a batch of data for training or validation.
        Args:
            split (str): 'train', 'val' or 'test' to specify which dataset to use.
        
        Returns:
            A batch of data containing input sequences and their corresponding target sequences.
        """
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val' or 'test'."

        ita_data = self.train_ita_data
        en_data = self.train_en_data
        
        if split == "val":
            ita_data = self.val_ita_data
            en_data = self.val_en_data
        elif split == "test":
            ita_data = self.test_ita_data
            en_data = self.test_en_data

        ita_batch = []
        en_batch_input = []
        en_batch_target = []
        for _ in range(self.batch_size):
            # The constructor checks if the "ita_data" size is equal to the "en_data" size.
            # We are sure that the two sets have the same size.
            idx = randint(0, len(ita_data) - 1)
            ita_sequence = ita_data[idx][:self.sequence_length]
            if len(ita_sequence) < self.sequence_length:
                ita_sequence = self.tokenizer.pad_sequence(ita_sequence, self.sequence_length)

            ita_batch.append(ita_sequence)

            en_sequence = en_data[idx][:self.sequence_length]
            if len(en_sequence) < self.sequence_length:
                en_sequence = self.tokenizer.pad_sequence(en_sequence, self.sequence_length)
            
            en_batch_input.append(en_sequence)

            en_target_sequence = en_data[idx][1:self.sequence_length + 1]
            if len(en_target_sequence) < self.sequence_length:
                en_target_sequence = self.tokenizer.pad_sequence(en_target_sequence, self.sequence_length)
            
            en_batch_target.append(en_target_sequence)

        ita_batch = torch.stack(ita_batch)
        en_batch_input = torch.stack(en_batch_input)
        en_batch_target = torch.stack(en_batch_target)

        return ita_batch, en_batch_input, en_batch_target
    
    def __len__(self):
        """
        Returns the number of batches in the dataset.
        Returns:
            int: Number of batches.
        """
        return (len(self.train_ita_data) + len(self.val_ita_data) + len(self.test_ita_data)) // self.batch_size
    

        
