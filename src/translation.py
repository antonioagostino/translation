import torch
from tokenizers.base import CharacterTokenizer
from encoders.text import SimpleTextEncoder
from modules.models import TranslationTransformer
from utils.parser import parse_translation_args

if __name__ == "__main__":

    # Parse command line arguments
    args = parse_translation_args()

    model_config_path = args.model_config
    checkpoint_path = args.checkpoint_path
    vocab_path = args.vocab_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharacterTokenizer(vocabulary_map=vocab_path)
    model = TranslationTransformer.create_from_yaml(model_config_path,
                                                    text_encoder=SimpleTextEncoder,
                                                    vocab_size=tokenizer.vocab_size,
                                                    device=device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sequence_to_translate = input("Digit the sentence you want to translate: ")
    input_tokens = tokenizer.tokenize(tokenizer.add_delimiters(sequence_to_translate))
    if len(input_tokens) < model.sequence_length:
        input_tokens = tokenizer.pad_sequence(input_tokens, model.sequence_length)
    else:
        input_tokens = input_tokens[:model.sequence_length]
    
    translated_sequence = model.translate(
        input_sequence=input_tokens,
        tokenizer=tokenizer,
    )

    print("Translated sequence:", tokenizer.detokenize(translated_sequence[1:-1]))
    