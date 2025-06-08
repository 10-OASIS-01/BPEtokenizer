# BPE Tokenizer

This project implements a tokenizer based on the Byte Pair Encoding (BPE) algorithm, with additional custom tokenizers, including one similar to the GPT-4 tokenizer. The implementation is inspired by the [minbpe](https://github.com/karpathy/minbpe/tree/master?tab=readme-ov-file) project by Andrej Karpathy. Some portions of the code are adapted from this project, which provides an efficient and minimalistic approach to BPE tokenization. The goal is to create a flexible tokenizer that can handle both small-scale datasets and scale to larger ones like OpenWebText or WikiText-103, while closely approximating the tokenization behavior seen in models like GPT-4.

## Table of Contents

- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Tokenizer](#training-the-tokenizer)
  - [Testing the Tokenizer](#testing-the-tokenizer)
- [License](#license)

## Project Structure

```
BPEtokenizer/
├── data/
│   ├── taylorswift.txt         (Source from Taylor Swift wiki for small-scale training and testing)
│   └── Salesforce___wikitext    (WikiText-2-raw-v1 dataset used for training)
├── models/
│   └── (Trained tokenizer models)
├── tokenizer/
│   ├── __init__.py              (Initializes the tokenizer module)
│   ├── helper.py               (Helper functions used across tokenizers)
│   ├── base.py                 (Base class for tokenizers with common methods)
│   ├── basic.py                (Implements a basic BPE tokenizer)
│   ├── regex.py                (Extends BPE with regex and special token handling)
│   └── gpt4.py                 (Wrapper for GPT-4 tokenizer functionality)
├── README.md
├── requirements.txt
├── train_tokenizer.py          (Script to train the tokenizer)
└── inference.py                (Script to test the tokenizer)
```

### File Descriptions

- `tokenizer/__init__.py`: Initializes the tokenizer module by importing the relevant tokenizer classes (`Tokenizer`, `BasicTokenizer`, `RegexTokenizer`, `GPT4Tokenizer`).
- `tokenizer/helper.py`: Contains helper functions used across different tokenizers, including functions for statistics gathering (`get_stats`), BPE merge operations (`merge`), character replacement (`replace_control_characters`), and encoding/decoding operations (`bpe`, `recover_merges`).
- `tokenizer/base.py`: Defines the base `Tokenizer` class with common methods for training, encoding, decoding, saving, and loading tokenizers. It relies on helper functions from `helper.py`.
- `tokenizer/basic.py`: Implements a minimal Byte Pair Encoding (BPE) tokenizer (`BasicTokenizer`) without regex splitting or special tokens. It extends the base `Tokenizer` class from `base.py`.
- `tokenizer/regex.py`: Extends the basic BPE tokenizer to handle regular expression-based token splitting patterns and special tokens (`RegexTokenizer`). It extends the `Tokenizer` class from `base.py`.
- `tokenizer/gpt4.py`: Implements a GPT-4-like tokenizer (`GPT4Tokenizer`) as a wrapper around the `RegexTokenizer`, using pre-trained merge files and handling special tokens specific to GPT-4. It extends the `RegexTokenizer` class from `regex.py`.

Note: The `RegexTokenizer` is the core tokenizer used for training, while the `GPT4Tokenizer` is a specialized implementation designed for comparison and testing purposes. It uses pretrained merges from the `cl100k_base` tokenizer, similar to GPT-4's tokenizer.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/10-OASIS-01/BPEtokenizer.git
    cd BPEtokenizer
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Tokenizer

The `train_tokenizer.py` script is used to train the tokenizer. It loads the WikiText-2-raw-v1 dataset, trains the tokenizer, and saves the trained model. For larger-scale datasets, such as OpenWebText or WikiText-103-raw-v1, training the tokenizer with a vocabulary size of 100,000 can help replicate the tokenization behavior seen in models like GPT-4.

#### Parameters

- `--data-dir`: Directory containing the training dataset (default: `data`)
- `--vocab-size`: Vocabulary size (default: `100000`)
- `--save-dir`: Directory to save the trained tokenizer model (default: `models`)
- `--model-name`: Name of the saved tokenizer model (default: `wikitext_tokenizer`)

#### Example

```sh
python train_tokenizer.py \
  --data-dir "data" \
  --vocab-size 32768 \
  --save-dir "models" \
  --model-name "wikitext_tokenizer" 
```

By training with larger datasets and a larger vocabulary, the tokenizer is likely to more closely approximate the behavior of tokenizers used in models like GPT-4.

### Testing the Tokenizer

The `inference.py` script contains various tests to ensure the tokenizer's encoding and decoding consistency. It checks for:

- Special tokens (e.g., `<|endoftext|>`, `<|fim_prefix|>`, etc.)
- Multilingual text
- Formatted text (e.g., bold, italics)
- Repeated patterns
- Long text
- Code snippets

#### Running the Tests

To run the tests, simply execute the `inference.py` script:

```sh
python inference.py
```

## Limitations

- **SentencePiece Tokenizer**: Currently, the tokenizer is based on the Byte Pair Encoding (BPE) algorithm. A future enhancement is planned to implement a tokenizer based on the SentencePiece algorithm, which is widely used in models like Llama and Mistral. This would expand the tokenizer's compatibility with more recent architectures.
  
- **Parallel Training**: The training code currently operates in a single-threaded manner, which can be slow for large datasets. There is potential for improving the training efficiency by introducing parallelism, such as using distributed training methods or leveraging multi-core processing to speed up tokenization on large-scale datasets.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


