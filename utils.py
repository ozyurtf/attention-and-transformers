import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, batch_size):
      super(PatchEmbedding, self).__init__()
      num_of_patches = (image_size * image_size)/(patch_size * patch_size)

      self.conv = nn.Conv2d(in_channels = in_channels,
                            out_channels = embed_dim,
                            kernel_size = (patch_size, patch_size),
                            stride = (patch_size, patch_size))
      self.flatten = nn.Flatten(2)
      self.class_token_embedding = nn.Parameter(torch.randn(batch_size, 1, embed_dim))
      self.position_embedding = nn.Parameter(torch.randn(batch_size, num_patches + 1, embed_dim))

    def forward(self, x):
      x = self.conv(x)
      x = self.flatten(x).transpose(1, 2)
      x = torch.cat([self.class_token_embedding, x], dim = 1)
      x = x + self.position_embedding

      return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
      super(MultiHeadSelfAttention, self).__init__()
      self.keys = nn.Linear(embed_dim, embed_dim)
      self.queries = nn.Linear(embed_dim, embed_dim)
      self.values = nn.Linear(embed_dim, embed_dim)
      self.multi_head_attention = nn.MultiheadAttention(embed_dim,
                                                        num_heads,
                                                        batch_first = True)

    def forward(self, x):
      k = self.keys(x)
      q = self.queries(x)
      v = self.values(x)
      x = self.multi_head_attention(k, q, v)
      return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, num_classes):
      super(TransformerBlock, self).__init__()
      # Layer Normalization
      self.layer_norm1 = nn.LayerNorm(embed_dim)

      # Multi Head Attention
      self.multi_head_attention1 = MultiHeadSelfAttention(embed_dim, num_heads)

      # Dropout
      self.dropout = nn.Dropout(dropout)

      # Layer Normalization
      self.layer_norm2 = nn.LayerNorm(embed_dim)

      # MLP
      self.linear1 = nn.Linear(embed_dim, mlp_dim)
      self.gelu = nn.GELU()
      self.linear2 = nn.Linear(mlp_dim, embed_dim)

      # Layer Normalization
      self.layer_norm3 = nn.LayerNorm(embed_dim)

      # Classification Layer
      self.linear3 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
      # Layer Normalization
      i1 = self.layer_norm1(x)

      # Multi Head Attention
      i2, _ = self.multi_head_attention1(i1)

      # Dropout
      i2 = self.dropout(i2)

      # Residual Connection
      i3 = i2 + x

      # Layer Normalization
      i4 = self.layer_norm2(i3)

      # MLP
      i5 = self.linear1(i4)
      i6 = self.gelu(i5)
      i7 = self.linear2(i6)

      # Residual Connection
      i8 = i7 + i3

      # Taking the average of the representations of all the patches
      i9 = torch.einsum('bne->be', i8) / i8.size(1)

      # Layer Normalization
      i10 = self.layer_norm3(i9)

      # Classification Layer
      i11 = self.linear3(i10)

      return i11

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size,
                 in_channels, embed_dim, num_heads,
                 mlp_dim, num_layers, num_classes,
                 batch_size, dropout=0.1):
      super(VisionTransformer, self).__init__()
      self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, batch_size)
      self.transformer_encoder = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, num_classes)

    def forward(self, x):
      x = self.patch_embedding(x)
      x = self.transformer_encoder(x)
      return x
