from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tokenizers.base import CharacterTokenizer

def parse_args():
    parser = ArgumentParser(description="Prepare EuroParl dataset for translation task.")
    parser.add_argument("--train-data-path", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--val-data-path", type=str, required=True, help="Path to the validation data CSV file.")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--tokenizer-vocab-path", type=str, required=True, help="Path to the tokenizer vocabulary map.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the prepared dataset.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path
    tokenizer_vocab_path = args.tokenizer_vocab_path
    output_path = args.output_path

    tokenizer = CharacterTokenizer(vocabulary_map=tokenizer_vocab_path)
    print("Loading dataset...")
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


    print("Starting tokenization...")
    train_ita_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in train_italian_sentences]
    train_en_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in train_english_sentences]
    val_ita_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in val_italian_sentences]
    val_en_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in val_english_sentences]
    test_ita_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in test_italian_sentences]
    test_en_data = [" ".join(list(map(str, tokenizer.tokenize(tokenizer.add_delimiters(sequence)).tolist()))) for sequence in test_english_sentences]

    print("Saving to file...")
    train_dataframe = {"sent_it": train_ita_data, "sent_en": train_en_data}
    val_dataframe = {"sent_it": val_ita_data, "sent_en": val_en_data}
    test_dataframe = {"sent_it": test_ita_data, "sent_en": test_en_data}

    pd.DataFrame(train_dataframe).to_csv(f"{output_path}/train.csv")
    pd.DataFrame(val_dataframe).to_csv(f"{output_path}/validation.csv")
    pd.DataFrame(test_dataframe).to_csv(f"{output_path}/test.csv")