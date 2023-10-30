import torch
import torch.nn as nn
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, batch_size):
      super(PatchEmbedding, self).__init__()
      self.conv = nn.Conv2d(in_channels = in_channels,
                            out_channels = embed_dim,
                            kernel_size = (patch_size, patch_size),
                            stride = (patch_size, patch_size),
                            padding = (patch_size, patch_size))

      self.relu = nn.ReLU()
      self.maxpool = nn.MaxPool2d(kernel_size=patch_size, stride=1)
      self.batch_norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
      x = self.conv(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.batch_norm(x)
      return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads = 8, dropout = 0.01):
        super().__init__()
        self.num_heads = num_heads

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(embed_dim, num_heads * 3, bias = False)

        self.to_out = nn.Sequential(nn.Linear(num_heads, embed_dim),
                                    nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2))

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, num_classes):
      super(TransformerBlock, self).__init__()

      # Layer Normalization
      self.layer_norm1 = nn.LayerNorm(embed_dim)

      # Multi Head Attention
      self.multi_head_attention1 = MultiHeadSelfAttention(embed_dim, num_heads)

      # Layer Normalization
      self.layer_norm2 = nn.LayerNorm(embed_dim)

      # MLP
      self.linear1 = nn.Linear(embed_dim, mlp_dim)
      self.gelu = nn.GELU()
      self.linear2 = nn.Linear(mlp_dim, embed_dim)

      # Initialize weights
      self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def forward(self, x):

      # Layer Normalization
      i1 = self.layer_norm1(x)

      # Multi Head Attention
      i2 = self.multi_head_attention1(i1)

      # Residual Connection
      i3 = i2 + x

      # MLP
      i5 = self.linear1(i3)
      i6 = self.gelu(i5)
      i7 = self.linear2(i6)

      # Residual Connection
      i8 = i7 + i3

      return i8

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size,
                 in_channels, embed_dim, num_heads,
                 mlp_dim, num_layers, num_classes,
                 batch_size, dropout=0.1):

      super(VisionTransformer, self).__init__()

      self.patch_embedding = PatchEmbedding(image_size, patch_size,
                                            in_channels, embed_dim, batch_size)

      self.flatten = nn.Flatten(2)

      self.class_token_embedding = nn.Parameter(torch.randn(batch_size, 1, embed_dim))


      self.position_embedding = nn.Parameter(torch.randn(batch_size,
                                                         50,
                                                         embed_dim))

      self.layers = nn.ModuleList([])

      for _ in range(num_layers):
       self.layers.append(nn.ModuleList([
           TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, num_classes)
           ]))

      # self.layer_norm3 = nn.LayerNorm(embed_dim)
      self.linear3 = nn.Linear(embed_dim, num_classes)
      self.identity = nn.Identity()

    def forward(self, x):
      x = self.patch_embedding(x)
      x = self.flatten(x)
      x = x. transpose(1, 2)
      x = torch.cat([self.class_token_embedding, x], dim = 1)
      x = x + self.position_embedding

      for transformer_encoder in self.layers:
         x = transformer_encoder[0](x)
         
      x = x.mean(dim = 1)
      x = self.linear3(x)
      x = self.identity(x)
      return x
