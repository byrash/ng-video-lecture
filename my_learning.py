import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 32  # how many independet sequences will we process in parallel
block_size = 8  # what is the maximum context length of the predictions
max_iterations = 5000  # how many iterations will we train for
eval_interval = 300  # how often will we evaluate the model
learning_rate = 1e-3  # what is the learning rate
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use the GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # use MAC GPU if available
eval_iters = 200  # how many iterations will we use for evaluation
n_embd = 32  # No of embedding dimensions
n_head = 4
n_layer = 4
dropout = 0.2
# Path to save/load the model
model_path = "bigram_language_model_complete.pth"
# -----------------

torch.manual_seed(1337)

with open('input.txt', 'r') as file:
    text = file.read()

# here are all the uniqe chacrters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from charcters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]  # encode: takes a string, outputs a list of integrs
def decode(l): return ''.join([itos[i] for i in l])  # decode: takes a list of integers, outputs a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% of the data will be train , rest validation
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C
        # Compute attention scored ("addinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ ( B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask out the upper half of the matrix
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projections = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projections(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_fwd = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layernorm1(x))
        x = x + self.feed_fwd(self.layernorm2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm_function = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tesnor of integers
        # B = Batch T = Time C= Channel(Vocab size)
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        x = self.blocks(x)  # Apply one head of self-attention. (B,T,C)
        x = self.layer_norm_function(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,Voacb size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print("B ", B, "T ", T, "C ", C)
            logits = logits.view(B*T, C)
            # print("logits shape ", logits.shape)
            # print("logits ", logits)
            targets = targets.view(B*T)
            # print("targets shape ", targets.shape)
            # print("targets ", targets)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes ( B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # becomes (B,1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # becomes (B, T+1)
        return idx


# Add the class to the safe globals list
torch.serialization.add_safe_globals([BigramLanguageModel])

# Check if the model file exists
if os.path.exists(model_path):
    # If model exists, load the model and optimizer state
    print("Loading pre-trained model...")
    model = torch.load(model_path, weights_only=False)  # Load the entire model if you saved the complete model
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Optionally, load the optimizer state if you saved it
    optimizer.load_state_dict(torch.load("optimizer.pth"))
else:
    # If model doesn't exist, initialize a new model
    print("Initializing a new model...")
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iterations):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'Iter {iter:5d}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')

        # Sample a batch of data (you need to implement this function)
        xb, yb = get_batch('train')  # Replace with your actual batch loading code

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # After training, save the model
    torch.save(model, model_path)  # Save the entire model (including architecture and parameters)
    # Optionally, save optimizer state
    torch.save(optimizer.state_dict(), "optimizer.pth")

# Function for loss estimation (just an example, implement this as needed)


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
