import os
import pickle
import torch
import argparse
from torch.nn import functional as F
from dotenv import load_dotenv

from model_lstm import LSTMModel
from model_transformer import TransformerModel

def generate_text(model, model_type, start_text, max_new_tokens, temperature, stoi, itos, block_size):
    """
    Autoregressive text generation.
    """
    model.eval()
    
    # Convert string to tensor
    context = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0)
    
    # For LSTM, we need to maintain the hidden state across generations
    hidden = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Transformers have a strict context window (block_size). 
            # If our sequence gets too long, we must crop it to the last block_size characters.
            if model_type == 'transformer':
                context_cropped = context[:, -block_size:]
                logits, _ = model(context_cropped, hidden=None)
            else:
                # LSTMs process the whole sequence, but we can just feed the last character 
                # if we keep track of the hidden state
                if context.shape[1] == 1 or hidden is None:
                    logits, hidden = model(context, hidden)
                else:
                    logits, hidden = model(context[:, -1:], hidden)
            
            # Focus only on the very last time step (the predicted next character)
            logits = logits[:, -1, :] 
            
            # Apply Temperature Scaling
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1)

    # Convert tensor back to string
    generated_chars = [itos[i.item()] for i in context[0]]
    return ''.join(generated_chars)

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained models")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--seed_text', type=str, default="ROMEO:")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=200)
    args = parser.parse_args()

    load_dotenv()
    BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 64))
    EMBED_SIZE = int(os.getenv("EMBED_SIZE", 128))
    HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 256))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", 3))
    NUM_HEADS = int(os.getenv("NUM_HEADS", 4))

    # Load Vocab Metadata
    meta_path = os.path.join("results", "meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    stoi = meta['stoi']
    itos = meta['itos']

    # Initialize Model
    if args.model == 'lstm':
        model = LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    else:
        model = TransformerModel(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE)

    # Load Weights
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    print(f"\n--- Generating with {args.model.upper()} (Temp: {args.temperature}) ---")
    print(f"Seed: '{args.seed_text}'\n")
    
    generated_output = generate_text(
        model=model, 
        model_type=args.model, 
        start_text=args.seed_text, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temperature, 
        stoi=stoi, 
        itos=itos, 
        block_size=BLOCK_SIZE
    )
    
    print(generated_output)
    print("\n---------------------------------------------------------")

if __name__ == '__main__':
    main()