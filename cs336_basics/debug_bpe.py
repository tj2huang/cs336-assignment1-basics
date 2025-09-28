from cs336_basics.train_bpe import train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def main():
    # Use the corpus.en file directly
    input_path = FIXTURES_PATH / "corpus.en"
    vocab_size = 500  # Start small for debugging
    special_tokens = ["<|endoftext|>"]

    print("Starting BPE training...")
    print(f"Input file: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    try:
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        print(f"Success! Vocabulary size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        print("First 10 merges:", merges[:10])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()