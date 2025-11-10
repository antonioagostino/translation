import torch
from datasets.europarl import EuroParlTranslationDataset
from tokenizers.base import CharacterTokenizer
from encoders.text import SimpleTextEncoder
from modules.models import TranslationTransformer
from utils.parser import parse_training_args


if __name__ == "__main__":

    # Parse command line arguments
    args = parse_training_args()

    ### Training parameters ###
    batch_size = args.batch_size
    iterations = args.iterations
    evaluation_interval = args.evaluation_interval
    evaluation_iterations = args.evaluation_iterations
    learning_rate = args.learning_rate

    sequence_length = args.sequence_length
    model_config_path = args.model_config
    vocab_path = args.vocab_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharacterTokenizer(vocabulary_map=vocab_path)
    print("Loading dataset...")
    dataset = EuroParlTranslationDataset(train_data_path="data/datasets/europarl_en_it_tokenized/train.csv",
                                         val_data_path="data/datasets/europarl_en_it_tokenized/validation.csv",
                                         test_data_path="data/datasets/europarl_en_it_tokenized/test.csv",
                                         sequence_length=sequence_length,
                                         batch_size=batch_size,
                                         tokenizer=tokenizer,
                                         already_tokenized=True)
    print("Dataset loaded.")
    model = TranslationTransformer.create_from_yaml(model_config_path,
                                                    text_encoder=SimpleTextEncoder,
                                                    vocab_size=tokenizer.vocab_size,
                                                    device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_loss_history, val_loss_history = model.fit(dataset=dataset,
                                                        optimizer=optimizer,
                                                        iterations=iterations,
                                                        evaluation_interval=evaluation_interval,
                                                        evaluation_iterations=evaluation_iterations,
                                                        best_model_path="best_model.pth",
                                                        log_losses_on_file=True)
    print("Training completed.")
    
