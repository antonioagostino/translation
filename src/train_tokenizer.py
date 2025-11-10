from tokenizers.base import CharacterTokenizer
from utils.parser import parse_tokenizer_args
import pandas as pd

if __name__ == "__main__":
    args = parse_tokenizer_args()
    save_path = args.save_path

    # Load dataset and prepare corpus
    train_dataframe = pd.read_csv("data/datasets/europarl_en_it/train.csv")
    train_italian_sentences = train_dataframe["sent_it"].to_list()
    train_english_sentences = train_dataframe["sent_en"].to_list()
    val_dataframe = pd.read_csv("data/datasets/europarl_en_it/validation.csv")
    val_italian_sentences = val_dataframe["sent_it"].to_list()
    val_english_sentences = val_dataframe["sent_en"].to_list()
    test_dataframe = pd.read_csv("data/datasets/europarl_en_it/validation.csv")
    test_italian_sentences = val_dataframe["sent_it"].to_list()
    test_english_sentences = val_dataframe["sent_en"].to_list()
    corpus = "\n".join(train_italian_sentences + train_english_sentences + val_italian_sentences + val_english_sentences + test_italian_sentences + test_english_sentences)
    
    tokenizer = CharacterTokenizer.train_tokenizer(corpus)
    tokenizer.save_vocabulary_map(save_path)
    