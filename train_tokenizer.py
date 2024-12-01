import os
import logging
import argparse
from datasets import load_dataset
from tokenizer import RegexTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define parameters directly in the code
DATASET_NAME = "wikitext-2-raw-v1"
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = {
    '<|endoftext|>': 32768,
    '<|fim_prefix|>': 32769,
    '<|fim_middle|>': 32770,
    '<|fim_suffix|>': 32771,
    '<|endofprompt|>': 32772
}

# Load the Wikitext dataset
def load_wikitext(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Use cache_dir to specify the cache directory, instead of data_dir as part of the configuration
    cache_dir = data_dir
    data_path = os.path.join(cache_dir, DATASET_NAME)

    # If the dataset exists, load it, otherwise redownload it
    if os.path.exists(data_path):
        logger.info(f"Dataset already exists, loading from: {data_path}")
        ds = load_dataset("Salesforce/wikitext", DATASET_NAME, cache_dir=cache_dir)
    else:
        logger.info(f"Downloading dataset to: {data_path}")
        ds = load_dataset("Salesforce/wikitext", DATASET_NAME, download_mode="force_redownload", cache_dir=cache_dir)

    # Return the text data from the training set
    return "\n".join(ds['train']['text'])


# Train the Tokenizer
def train_tokenizer(train_text, vocab_size=100000, pattern=None, special_tokens=None):
    """Train a tokenizer based on the provided text"""
    tokenizer = RegexTokenizer(pattern)
    tokenizer.train(train_text, vocab_size=vocab_size, verbose=False)
    # Register special tokens
    if special_tokens:
        tokenizer.register_special_tokens(special_tokens)

    return tokenizer


# Save the Tokenizer
def save_tokenizer(tokenizer, save_dir="models", model_name="wikitext_tokenizer"):
    """Save the trained tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, model_name)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved as {tokenizer_path}.model and {tokenizer_path}.vocab")


# Main function
def main(args):
    # Load the training text
    logger.info("Starting to load dataset...")
    train_text = load_wikitext(data_dir=args.data_dir)

    # Train the Tokenizer
    logger.info("Starting to train tokenizer...")
    tokenizer = train_tokenizer(
        train_text, vocab_size=args.vocab_size, pattern=SPLIT_PATTERN, special_tokens=SPECIAL_TOKENS)

    # Save the Tokenizer
    logger.info("Saving the trained tokenizer...")
    save_tokenizer(tokenizer, save_dir=args.save_dir, model_name=args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer on the WikiText dataset")
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory to store the dataset')
    parser.add_argument('--vocab-size', type=int, default=100000, help='Vocabulary size')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save the tokenizer model')
    parser.add_argument('--model-name', type=str, default='wikitext_tokenizer', help='Name of the tokenizer model')
    args = parser.parse_args()

    main(args)


"""
python train_tokenizer.py \
  --data-dir "data" \
  --vocab-size 32768 \
  --save-dir "models" \
  --model-name "wikitext_tokenizer" 
"""

"""
python train_tokenizer.py ^
  --data-dir "data" ^
  --vocab-size 32768 ^
  --save-dir "models" ^
  --model-name "wikitext_tokenizer"
"""
