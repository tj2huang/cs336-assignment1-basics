from cs336_basics.train_bpe import train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import cProfile
import pstats
import io
from pstats import SortKey

def main():
    # Use the corpus.en file directly
    # input_path = FIXTURES_PATH / "corpus.en"
    input_path = 'C:/Users/tomhu/data/tinystoriesv2-valid.txt'
    vocab_size = 10000 
    special_tokens = ["<|endoftext|>", "<|startoftext|>", "<|pad|>"]

    print("Starting BPE training...")
    print(f"Input file: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    try:
        # Profile the train_bpe function
        profiler = cProfile.Profile()
        profiler.enable()
        
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        
        profiler.disable()
        
        print(f"Success! Vocabulary size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        print("First 10 merges:", merges[:10])
        
        # Analyze profiling results
        print("\n" + "="*50)
        print("PROFILING RESULTS")
        print("="*50)
        
        # Create a string buffer to capture stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(20)  # Show top 20 functions by cumulative time
        print(s.getvalue())
        
        # Also show by total time
        print("\n" + "-"*30)
        print("TOP FUNCTIONS BY TOTAL TIME")
        print("-"*30)
        s2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=s2).sort_stats(SortKey.TOTAL_TIME)
        ps2.print_stats(15)
        print(s2.getvalue())
        
        # Save detailed profile to file
        profiler.dump_stats('train_bpe_profile.prof')
        print(f"\nDetailed profile saved to 'train_bpe_profile.prof'")
        print("You can analyze it further with: python -m pstats train_bpe_profile.prof")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()