import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """
        Initializes the LSTM model.
        
        Args:
            vocab_size (int): Number of unique characters in the dataset.
            embed_size (int): Dimension of the character embeddings.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Embedding Layer: Converts integer tokens into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. LSTM Layer: The core sequence processor
        # batch_first=True means our input tensors will have shape (Batch, Sequence, Feature)
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 3. Fully Connected Layer: Maps the LSTM hidden states back to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length)
            hidden (tuple, optional): Tuple containing (h_0, c_0) hidden states.
        
        Returns:
            logits (Tensor): Unnormalized predictions of shape (batch_size, sequence_length, vocab_size)
            hidden (tuple): The updated hidden states
        """
        # x shape: (B, T) where B is batch size, T is sequence length (block size)
        
        # Pass through embedding layer
        embed = self.embedding(x) 
        # embed shape: (B, T, C) where C is embed_size
        
        # Pass through LSTM
        # If hidden is None, PyTorch automatically initializes it to zeros
        out, hidden = self.lstm(embed, hidden)
        # out shape: (B, T, H) where H is hidden_size
        
        # Pass through linear layer to get raw scores (logits) for each character in vocab
        logits = self.fc(out)
        # logits shape: (B, T, V) where V is vocab_size
        
        return logits, hidden