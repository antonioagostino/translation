from argparse import ArgumentParser

def parse_training_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--iterations", type=int, default=500000, help="Number of training iterations.")
    parser.add_argument("--evaluation_interval", type=int, default=1000, help="Interval for evaluation during training.")
    parser.add_argument("--evaluation_iterations", type=int, default=1000, help="Number of iterations for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--sequence_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--model_config", type=str, default="./configs/transformers.yaml", help="Path to the model configuration YAML file.")
    parser.add_argument("--vocab_path", type=str, default="data/vocabularies/tokenizer.json", help="Path to the vocabulary file.")
    
    return parser.parse_args()

def parse_translation_args():
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, default="./configs/transformers.yaml", help="Path to the model configuration YAML file.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.pth", help="Path to the model checkpoint file.")
    parser.add_argument("--vocab_path", type=str, default="data/vocabularies/tokenizer.json", help="Path to the vocabulary file.")

    return parser.parse_args()

def parse_tokenizer_args():
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data/vocabularies/tokenizer.json", help="Path to save the trained tokenizer.")

    return parser.parse_args()