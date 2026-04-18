import os
import torch
import pickle
from dotenv import load_dotenv
import urllib.request

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "input/tinyshakespeare.txt")
RESULTS_DIR = "results"

def prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Downloading...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, DATA_PATH)
    
    # 1. Read the raw text
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset length: {len(text):,} characters")
    
    # 2. Build the Vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {''.join(chars)}")
    
    # 3. Create mapping dictionaries (stoi: string-to-int, itos: int-to-string)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode function: takes a string, outputs a list of integers
    encode = lambda s: [stoi[c] for c in s]
    
    # 4. Tokenize the entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # 5. Split into Training (90%) and Validation (10%) sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 6. Save artifacts to /results for the models to use
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    torch.save(train_data, os.path.join(RESULTS_DIR, 'train_data.pt'))
    torch.save(val_data, os.path.join(RESULTS_DIR, 'val_data.pt'))
    
    # Save the mappings so we can decode the text later during generation
    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }
    with open(os.path.join(RESULTS_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    print(f"Saved tokenized data and metadata to {RESULTS_DIR}/")

if __name__ == '__main__':
    prepare_data()