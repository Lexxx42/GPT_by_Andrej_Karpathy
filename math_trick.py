# The mathematical trick in self-attention
import torch
import torch.nn as nn
from torch.nn import functional as F

# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

# consider the following toy example:

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)

# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) ----> (B, T, C)
print(torch.allclose(xbow, xbow2))

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))

# version 4: self-attention!
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
# out = wei @ x

print(out.shape)
print(wei[0])

# Notes:

# Attention is a communication mechanism.
# Can be seen as nodes in a directed graph looking at each other and aggregating
# information with a weighted sum from all nodes that point to them,
# with data-dependent weights.

# There is no notion of space.
# Attention simply acts over a set of vectors.
# This is why we need to positionally encode tokens.

# Each example across batch dimension is of course processed completely
# independently and never "talk" to each other.

# In an "encoder" attention block just delete the single line
# that does masking with tril, allowing all tokens to communicate.
# This block here is called a "decoder" attention block
# because it has triangular masking, and is usually used in autoregressive settings,
# like language modeling.

# "Self-attention" just means that the keys and values are produced
# from the same source as queries.
# In "cross-attention", the queries still get produced from x,
# but the keys and values come from some other,
# external source (e.g. an encoder module)

# "Scaled" attention additional divides wei by 1/sqrt(head_size).
# This makes it so when input Q,K are unit variance,
# wei will be unit variance too and Softmax will stay diffuse and not saturate too much.
# Illustration below. (look at README.md)

k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5

print(k.var())

print(q.var())

print(wei.var())

print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1))

print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]) * 8, dim=-1))  # gets too peaky, converges to one-hot
