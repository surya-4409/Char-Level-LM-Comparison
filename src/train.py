import os
import json
import torch
import argparse
from torch.nn import functional as F
from dotenv import load_dotenv

# Import our custom models
from model_lstm import LSTMModel
from model_transformer import TransformerModel

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(description="Train Character-Level LM")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'], 
                        help="Choose which model to train: 'lstm' or 'transformer'")
    args = parser.parse_args()

    # 2. Load Environment Variables
    load_dotenv()
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 64))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
    EPOCHS = int(os.getenv("EPOCHS", 10))
    EVAL_ITERS = int(os.getenv("EVAL_ITERS", 100))
    EMBED_SIZE = int(os.getenv("EMBED_SIZE", 128))
    HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 256))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", 3))
    NUM_HEADS = int(os.getenv("NUM_HEADS", 4))
    
    RESULTS_DIR = "results"
    
    print(f"--- Starting Training for {args.model.upper()} ---")

    # 3. Load Tokenized Data
    train_data = torch.load(os.path.join(RESULTS_DIR, 'train_data.pt'))
    val_data = torch.load(os.path.join(RESULTS_DIR, 'val_data.pt'))
    
    import pickle
    with open(os.path.join(RESULTS_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']

    # 4. Batch Generation Function
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # Generate random starting indices for the batch
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        # Stack the context (x) and the target (y, shifted by 1)
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y

    # 5. Initialize the chosen model
    if args.model == 'lstm':
        model = LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    elif args.model == 'transformer':
        model = TransformerModel(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters.")

    # 6. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Training Loop
    loss_history = {'train': [], 'val': []}
    
    # We define an "epoch" here as a certain number of iterations
    # To keep CPU training fast, let's say 1 epoch = EVAL_ITERS batches
    total_iters = EPOCHS * EVAL_ITERS
    
    model.train()
    for iter_num in range(1, total_iters + 1):
        # Grab a batch
        xb, yb = get_batch('train')
        
        # Forward pass (hidden state starts as None for every new random batch)
        logits, _ = model(xb, hidden=None)
        
        # Reshape for CrossEntropy: (B, T, C) -> (B*T, C)
        B, T, C = logits.shape
        logits_reshaped = logits.view(B*T, C)
        targets_reshaped = yb.view(B*T)
        
        # Calculate Loss
        loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping to prevent exploding gradients (NaN loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Evaluation step
        if iter_num % EVAL_ITERS == 0 or iter_num == total_iters:
            # Switch to eval mode
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                val_logits, _ = model(xv, hidden=None)
                val_loss = F.cross_entropy(val_logits.view(B*T, C), yv.view(B*T))
            
            # Record losses
            current_epoch = iter_num // EVAL_ITERS
            train_loss_val = loss.item()
            val_loss_val = val_loss.item()
            
            loss_history['train'].append(train_loss_val)
            loss_history['val'].append(val_loss_val)
            
            print(f"Epoch {current_epoch}/{EPOCHS} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f}")
            
            # Switch back to train mode
            model.train()

    # 8. Save Artifacts
    model_path = os.path.join(RESULTS_DIR, f"{args.model}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    
    loss_path = os.path.join(RESULTS_DIR, f"{args.model}_loss.json")
    with open(loss_path, 'w') as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to {loss_path}")

if __name__ == '__main__':
    main()