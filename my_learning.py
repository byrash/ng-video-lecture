import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independet sequences will we process in parallel
block_size = 8  # what is the maximum context length of the predictions
max_iterations = 3000  # how many iterations will we train for
eval_interval = 300  # how often will we evaluate the model
learning_rate = 1e-2  # what is the learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use the GPU if available
eval_iters = 200  # how many iterations will we use for evaluation
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tesnor of integers
        # B = Batch T = Time C= Channel(Vocab size)
        logits = self.token_embedding_table(idx)  # (B,T,C)
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes ( B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # becomes (B,1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # becomes (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)  # vocab_size is length of all unique characters in input text data
m = model.to(device)

# create a PyTorch optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter:5d}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')

    # Sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
