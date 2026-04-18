# Character-Level Language Model Comparison
**Author:** Billakurti Venkata Suryanarayana (Roll No: 23MH1A4409)

### Perplexity Comparison
| Model | Final Validation Loss | Perplexity ($PPL = e^L$) |
| :--- | :--- | :--- |
| **LSTM** | 1.8265 | 6.2120 |
| **Mini-Transformer** | 1.7790 | 5.9237 |

### Qualitative Analysis
**1. Temperature Effects:**
* **Temperature 0.5:** Models produce highly repetitive, safe character sequences with very little creativity. It often gets stuck in loops.
* **Temperature 1.0:** The optimal balance. It captures Shakespearean formatting (Speaker tags in ALL CAPS) and generates basic English structures, though vocabulary is limited due to the short CPU training time.
* **Temperature 1.5:** Highly chaotic and random. The text largely degrades into gibberish as the probability distribution is flattened, giving low-likelihood characters an unfair chance to be selected.

**2. Architectural Differences:**
The Transformer achieved a lower validation loss and perplexity with significantly fewer parameters than the LSTM. By utilizing Multi-Head Self-Attention, the Transformer can look at the entire context window simultaneously to capture relationships, whereas the LSTM relies on a sequential hidden state that acts as an informational bottleneck.
