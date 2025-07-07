import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def visualize_attention(model, data_loader, idx2word, device, num_samples=3, max_tokens=100, pad_idx=0):
    model.eval()
    shown = 0

    with torch.no_grad():
        ## we will show the attention_weight plots on for one batch
        ## I think it will be enough to analyse how the model dynamically focuses on the most relevant parts of the input when generating outputs
        for texts, labels in data_loader:
            texts = texts.to(device)
            outputs, attn_weights = model(texts)

            ## if we didnt use attention mechanism for a particular model, we immediately return
            if attn_weights is None:
                print("This model does not use attention.")
                return

            for i in range(min(num_samples, len(texts))):
                token_indices = texts[i].cpu().numpy()
                weights = attn_weights[i].cpu().numpy()

                filtered = [(idx2word[tok], weight) for tok, weight in zip(token_indices, weights) if tok != pad_idx] # Filter out PAD tokens
                filtered = filtered[:max_tokens] ## only use some tokens so that we can visualise properly
                tokens, weights = zip(*filtered)

                plt.figure(figsize=(min(0.6 * len(tokens), 12), 2.5))
                sns.heatmap([weights], cmap='viridis', xticklabels=tokens, yticklabels=['Attention'], cbar=True, annot=np.array([["{:.2f}".format(w) for w in weights]]), fmt='', annot_kws={"rotation": 90, "fontsize": 9, "va": "center", "ha": "center"})                
                plt.title("Attention Weights")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

            break 
