# Another transformers exercise 
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    #https://paperswithcode.com/method/scaled
    # Query, key, value, divided by dimension so we have a variance of 1
    def __init__(self):
        super().__init__()
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):

        # Straightforward matmul
        attn = torch.matmul(q, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, 1e-9)

        attn = F.softmax(attn, dim = -1)
        # use the v cause nothing happened with that mat
        output = torch.matmul(attn, v)

        return output, attn
    

    