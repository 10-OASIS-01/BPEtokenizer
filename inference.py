from tokenizer import RegexTokenizer


def load_tokenizer(model_path):
    """Load the trained tokenizer model"""
    tokenizer = RegexTokenizer()
    try:
        tokenizer.load(model_path)
        print(f"Tokenizer model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading tokenizer model from {model_path}: {e}")
        exit(1)  # Stop execution if loading fails
    return tokenizer


def encode_decode_consistency(tokenizer, test_strings):
    """Test encode-decode consistency"""
    for text in test_strings:
        print(f"Testing encode-decode consistency for: {text}")

        # Encode text
        token_ids = tokenizer.encode(text)
        print(f"Encoded Token IDs: {token_ids}")

        # Decode token IDs
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded Text: {decoded_text}")

        # Check if encoding and decoding results are consistent
        assert text == decoded_text, f"Encoding and decoding for text '{text}' failed!"
        print("Encode-Decode Test Passed\n")


def special_tokens(tokenizer, special_string):
    """Test special token handling"""
    print("Testing special tokens handling...")

    # Encode special string
    special_token_ids = tokenizer.encode(special_string, allowed_special="all")
    print(f"Special Token IDs: {special_token_ids}")

    # Decode special token IDs
    decoded_special_string = tokenizer.decode(special_token_ids)
    print(f"Decoded Special String: {decoded_special_string}")

    # Check if the encoding and decoding of the special string are consistent
    assert special_string == decoded_special_string, "Special token handling failed!"
    print("Special Tokens Test Passed\n")


def multilingual_encode_decode(tokenizer, multilingual_strings):
    """Test multilingual character sets"""
    for text in multilingual_strings:
        print(f"Testing multilingual encode-decode consistency for: {text}")

        # Encode text
        token_ids = tokenizer.encode(text)
        print(f"Encoded Token IDs: {token_ids}")

        # Decode token IDs
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded Text: {decoded_text}")

        # Check if encoding and decoding results are consistent
        assert text == decoded_text, f"Encoding and decoding for text '{text}' failed!"
        print("Multilingual Encode-Decode Test Passed\n")


def format_specific_cases(tokenizer, formatted_strings):
    """Test common text formats"""
    for text in formatted_strings:
        print(f"Testing format-specific encode-decode consistency for: {text}")

        # Encode text
        token_ids = tokenizer.encode(text)
        print(f"Encoded Token IDs: {token_ids}")

        # Decode token IDs
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded Text: {decoded_text}")

        # Check if encoding and decoding results are consistent
        assert text == decoded_text, f"Encoding and decoding for formatted text '{text}' failed!"
        print("Format Test Passed\n")


def repeated_patterns(tokenizer, repeated_strings):
    """Test repeated patterns and whitespace characters"""
    for text in repeated_strings:
        print(f"Testing repeated and whitespace patterns encode-decode for: {text}")

        # Encode text
        token_ids = tokenizer.encode(text)
        print(f"Encoded Token IDs: {token_ids}")

        # Decode token IDs
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded Text: {decoded_text}")

        # Check if encoding and decoding results are consistent
        assert text == decoded_text, f"Encoding and decoding for repeated text '{text}' failed!"
        print("Repeated Pattern Test Passed\n")


def long_text_handling(tokenizer, long_text):
    """Test long text handling"""
    print("Testing long text handling...")

    # Encode long text
    long_token_ids = tokenizer.encode(long_text)
    print(f"Encoded Long Text Token IDs: {long_token_ids[:20]}...")  # Print the first 20 token IDs

    # Decode long text
    decoded_long_text = tokenizer.decode(long_token_ids)
    print(f"Decoded Long Text (First 500 characters): {decoded_long_text[:500]}...")  # Print the first 500 characters

    # Check if the encoding and decoding of the long text are consistent
    assert long_text == decoded_long_text, "Long text encoding and decoding failed!"
    print("Long Text Test Passed\n")


def code_text_handling(tokenizer, code_texts):
    """Test code text handling"""
    print("Testing code text handling...")

    for text in code_texts:
        print(f"Testing code text encode-decode consistency for:\n{text}\n")

        # Encode code text
        token_ids = tokenizer.encode(text)
        print(f"Encoded Code Token IDs: {token_ids[:20]}...")  # Print the first 20 token IDs

        # Decode code text
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded Code Text (First 500 characters): \n{decoded_text[:500]}...")  # Print the first 500 characters

        # Check if the encoding and decoding of the code text are consistent
        assert text == decoded_text, f"Code text encoding and decoding failed for:\n{text}"
        print("Code Text Test Passed\n")


def main():
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
    print("\n--- Testing encode-decode consistency ---")
    encode_decode_consistency(tokenizer, test_strings)

    print("\n--- Testing special tokens handling ---")
    special_tokens(tokenizer, special_string)

    print("\n--- Testing multilingual encode-decode consistency ---")
    multilingual_encode_decode(tokenizer, multilingual_strings)

    print("\n--- Testing format-specific encode-decode consistency ---")
    format_specific_cases(tokenizer, formatted_strings)

    print("\n--- Testing repeated patterns encode-decode ---")
    repeated_patterns(tokenizer, repeated_strings)

    print("\n--- Testing long text handling ---")
    long_text_handling(tokenizer, long_text)

    print("\n--- Testing code text handling ---")
    code_text_handling(tokenizer, code_texts)


if __name__ == "__main__":
    main()
