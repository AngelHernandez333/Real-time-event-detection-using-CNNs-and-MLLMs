from transformers import AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import math
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = 'Deep learning is simple.'
print(tokenizer(text, add_special_tokens=False, return_tensors='pt'))
input = tokenizer(text, add_special_tokens=False, return_tensors='pt')

vocab_size = 30522
hidden_size = 768
intermediate_size = 3072
num_heads = 12

token_embeds = nn.Embedding(vocab_size, hidden_size)
input_embeds = token_embeds(input.input_ids)
print(token_embeds) # output: Embedding(30522, 768)
print(input_embeds.size()) # output: torch.Size([1, 5, 768])

query = key = value = input_embeds

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    # bmm is batch matrix multiplication
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)

    # Skip masking
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_state):
        # Use scaled dot product attention
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = hidden_size // num_heads

        # Multiple heads
        self.heads = nn.ModuleList([AttentionHead(hidden_size, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state):
        # Concatenate the scaled dot product outputs 
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

multihead_attn = MultiHeadAttention()
attn_output = multihead_attn(input_embeds)
print(attn_output.size()) # output: torch.Size([1, 5, 768])

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, intermediate_size)
        self.lin2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
feed_forward = FeedForward()
ff_outputs = feed_forward(attn_output)
print(ff_outputs.size()) # output : torch.Size([1, 5, 768])
print(ff_outputs)

#https://www.aibutsimple.com/p/transformer-models-from-scratch-in-python?_bhlid=ab9eb4fa09ef52af5b006076e7bef67ab2246d99&utm_campaign=transformer-models-from-scratch-in-python&utm_medium=newsletter&utm_source=www.aibutsimple.com