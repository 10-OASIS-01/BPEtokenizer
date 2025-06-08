from tokenizer import RegexTokenizer

import sys

LOG_FILE = "inference.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def load_tokenizer(model_path):
    """Load the trained tokenizer model"""
    tokenizer = RegexTokenizer()
    try:
        tokenizer.load(model_path)
        log(f"Tokenizer model loaded from {model_path}")
        print(f"[INFO] Tokenizer model loaded from {model_path}")
    except Exception as e:
        log(f"Error loading tokenizer model from {model_path}: {e}")
        print(f"[ERROR] Error loading tokenizer model from {model_path}: {e}")
        sys.exit(1)  # Stop execution if loading fails
    return tokenizer

def encode_decode_consistency(tokenizer, test_strings):
    """Test encode-decode consistency"""
    print("[INFO] Running encode-decode consistency tests...")
    for text in test_strings:
        log(f"Testing encode-decode consistency for: {text}")
        token_ids = tokenizer.encode(text)
        log(f"Encoded Token IDs: {token_ids}")
        decoded_text = tokenizer.decode(token_ids)
        log(f"Decoded Text: {decoded_text}")
        assert text == decoded_text, f"Encoding and decoding for text '{text}' failed!"
        log("Encode-Decode Test Passed\n")
    print("[INFO] Encode-decode consistency tests finished.")

def special_tokens(tokenizer, special_string):
    """Test special token handling"""
    print("[INFO] Running special tokens tests...")
    log("Testing special tokens handling...")
    special_token_ids = tokenizer.encode(special_string, allowed_special="all")
    log(f"Special Token IDs: {special_token_ids}")
    decoded_special_string = tokenizer.decode(special_token_ids)
    log(f"Decoded Special String: {decoded_special_string}")
    assert special_string == decoded_special_string, "Special token handling failed!"
    log("Special Tokens Test Passed\n")
    print("[INFO] Special tokens tests finished.")

def multilingual_encode_decode(tokenizer, multilingual_strings):
    """Test multilingual character sets"""
    print("[INFO] Running multilingual encode-decode tests...")
    for text in multilingual_strings:
        log(f"Testing multilingual encode-decode consistency for: {text}")
        token_ids = tokenizer.encode(text)
        log(f"Encoded Token IDs: {token_ids}")
        decoded_text = tokenizer.decode(token_ids)
        log(f"Decoded Text: {decoded_text}")
        assert text == decoded_text, f"Encoding and decoding for text '{text}' failed!"
        log("Multilingual Encode-Decode Test Passed\n")
    print("[INFO] Multilingual encode-decode tests finished.")

def format_specific_cases(tokenizer, formatted_strings):
    """Test common text formats"""
    print("[INFO] Running format-specific encode-decode tests...")
    for text in formatted_strings:
        log(f"Testing format-specific encode-decode consistency for: {text}")
        token_ids = tokenizer.encode(text)
        log(f"Encoded Token IDs: {token_ids}")
        decoded_text = tokenizer.decode(token_ids)
        log(f"Decoded Text: {decoded_text}")
        assert text == decoded_text, f"Encoding and decoding for formatted text '{text}' failed!"
        log("Format Test Passed\n")
    print("[INFO] Format-specific encode-decode tests finished.")

def repeated_patterns(tokenizer, repeated_strings):
    """Test repeated patterns and whitespace characters"""
    print("[INFO] Running repeated patterns encode-decode tests...")
    for text in repeated_strings:
        log(f"Testing repeated and whitespace patterns encode-decode for: {text}")
        token_ids = tokenizer.encode(text)
        log(f"Encoded Token IDs: {token_ids}")
        decoded_text = tokenizer.decode(token_ids)
        log(f"Decoded Text: {decoded_text}")
        assert text == decoded_text, f"Encoding and decoding for repeated text '{text}' failed!"
        log("Repeated Pattern Test Passed\n")
    print("[INFO] Repeated patterns encode-decode tests finished.")

def long_text_handling(tokenizer, long_text):
    """Test long text handling"""
    print("[INFO] Running long text handling test...")
    log("Testing long text handling...")
    long_token_ids = tokenizer.encode(long_text)
    log(f"Encoded Long Text Token IDs: {long_token_ids[:20]}...")  # Print the first 20 token IDs
    decoded_long_text = tokenizer.decode(long_token_ids)
    log(f"Decoded Long Text (First 500 characters): {decoded_long_text[:500]}...")  # Print the first 500 characters
    assert long_text == decoded_long_text, "Long text encoding and decoding failed!"
    log("Long Text Test Passed\n")
    print("[INFO] Long text handling test finished.")

def code_text_handling(tokenizer, code_texts):
    """Test code text handling"""
    print("[INFO] Running code text handling tests...")
    log("Testing code text handling...")
    for text in code_texts:
        log(f"Testing code text encode-decode consistency for:\n{text}\n")
        token_ids = tokenizer.encode(text)
        log(f"Encoded Code Token IDs: {token_ids[:20]}...")  # Print the first 20 token IDs
        decoded_text = tokenizer.decode(token_ids)
        log(f"Decoded Code Text (First 500 characters): \n{decoded_text[:500]}...")  # Print the first 500 characters
        assert text == decoded_text, f"Code text encoding and decoding failed for:\n{text}"
        log("Code Text Test Passed\n")
    print("[INFO] Code text handling tests finished.")

def main():
    print("[INFO] Inference script started.")
    # Clear log file at the start
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    # Load the tokenizer model
    model_path = "models/wikitext_tokenizer.model"
    tokenizer = load_tokenizer(model_path)

    # Test cases: strings containing various scenarios
    test_strings = [
        "",  # Empty string
        "?",  # Single character
        "hello world!!!? (안녕하세요!) lol123 😉",  # Mixed characters
        "FILE:taylorswift.txt",  # Special handling of file paths
        "你好，世界！This is a test.",  # Mixed Chinese and English
        "Testing punctuation! @#$%^&*()_+",  # Special symbols
        "Test\tTab and\nNewline characters.",  # Special characters
        "This is a very long string " * 20,  # Long string
    ]

    # Test special strings: containing various special tokens (e.g., <|endoftext|>)
    special_string = """
    <|endoftext|>Hello world this is one document
    <|endoftext|>And this is another document
    <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
    <|endoftext|>Last document!!! 👋<|endofprompt|>
    """

    # Test multilingual string set: including English, Japanese, Korean, Arabic, etc.
    multilingual_strings = [
        "Hello, world!",  # English
        "こんにちは、世界！",  # Japanese
        "안녕하세요, 세계!",  # Korean
        "مرحبا بالعالم!",  # Arabic
        "Привет, мир!",  # Russian
        "Γειά σας, κόσμος!",  # Greek
        "नमस्ते दुनिया!",  # Hindi
        "¡Hola, mundo!",  # Spanish
        "😊🌍🚀",  # Emojis
    ]

    # Test formatted text: URL, email addresses, HTML tags, etc.
    formatted_strings = [
        "Visit https://www.example.com for more info.",  # URL
        "Contact me at email@example.com.",  # Email address
        "Call me at (123) 456-7890.",  # Phone number
        "<html><body><h1>Hello World</h1></body></html>",  # HTML tags
        "The quick brown fox jumps over the lazy dog.",  # Common English sentence
    ]

    # Test repeated patterns and whitespace characters: including repeated words, spaces, tabs, etc.
    repeated_strings = [
        "word word word word",  # Repeated words
        "  space  space  ",  # Repeated spaces
        "line1\nline2\nline3",  # Multi-line text
        "\tTab\tTab\t",  # Tabs
        "!!!@@@###$$$",  # Repeated special symbols
    ]

    # Test code text: simulate common code snippets
    code_texts = [
        # Python example
        """def hello_world():
    print("Hello, world!")""",

        # JavaScript example
        """function greet() {
    console.log('Hello, world!');
}""",

        # C example
        """#include <stdio.h>

int main() {
    printf("Hello, world!");
    return 0;
}""",

        # HTML example
        """<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>""",

        # Java example
        """public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}"""
    ]

    # Generate a very long text for testing long text handling
    long_text = "This is a very long document. " * 1000  # Generate a very long text

    # Perform all tests
    log("\n--- Testing encode-decode consistency ---")
    encode_decode_consistency(tokenizer, test_strings)

    log("\n--- Testing special tokens handling ---")
    special_tokens(tokenizer, special_string)

    log("\n--- Testing multilingual encode-decode consistency ---")
    multilingual_encode_decode(tokenizer, multilingual_strings)

    log("\n--- Testing format-specific encode-decode consistency ---")
    format_specific_cases(tokenizer, formatted_strings)

    log("\n--- Testing repeated patterns encode-decode ---")
    repeated_patterns(tokenizer, repeated_strings)

    log("\n--- Testing long text handling ---")
    long_text_handling(tokenizer, long_text)

    log("\n--- Testing code text handling ---")
    code_text_handling(tokenizer, code_texts)

    print("[INFO] All inference tests finished. See inference.log for details.")

if __name__ == "__main__":
    main()
