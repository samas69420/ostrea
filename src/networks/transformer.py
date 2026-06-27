import torch
import torch.nn.functional as F


def extract_patches(x, patch_size):
    """
    (C, H, W) -> (num_patches, patch_size^2 * C)
    (B, C, H, W) -> (B, num_patches, patch_size^2 * C)
    """
    _, _, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    is_unbatched = x.dim() == 3

    if is_unbatched:
        x = x.unsqueeze(0)
        
    B, C, H, W = x.shape
    
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"input dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})")

    # split into patches -> [B, C, H//p, p, W//p, p]
    patches = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    
    # rearrange dimensions to group patches in the right way -> [B, H//p, W//p, C, p, p]
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    
    # flatten patches cause the transformer wants a sequence of 1d-vectors -> [B, num_patches, C * p^2]
    patches = patches.reshape(B, -1, C * patch_size * patch_size)
    
    if is_unbatched:
        return patches.squeeze(0)
    return patches


class PositionalEncoding(torch.nn.Module):

    def __init__(self, model_size, max_seq_len = 1000):
        """
        learned absolute positional encoding
        """

        super().__init__()
        
        self.positional_encoding_matrix = torch.nn.Parameter(torch.randn(max_seq_len,model_size))

    def forward(self, x):

        batch, seq_len, *_ = x.shape

        return x + self.positional_encoding_matrix[:seq_len].unsqueeze(0)


class MultiHeadAttention(torch.nn.Module):
    """
    efficient version of MHA that doesn't instantiate every single head and loop 
    over all of them but process them in parallel, always expect batched input
    """

    def __init__(self, model_size, n_heads, head_dim):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.model_size = model_size

        # single large Linear layer for all Q, K, V projections for all heads
        self.W_qkv = torch.nn.Linear(model_size, 3 * n_heads * head_dim, bias=False)
        
        self.W_o = torch.nn.Linear(n_heads * head_dim, model_size, bias=False)

    def forward(self, x):

        # x shape: [batch_size, seq_len, model_size]
        batch_size, seq_len, _ = x.shape

        # qkv shape: [batch_size, seq_len, 3 * n_heads * head_dim]
        qkv = self.W_qkv(x)

        # without considering batch we have that:
        #              _                                             _ 
        #             |               |               |               |
        # X * W_qkv = | Q1...Qn_heads | K1...Kn_heads | V1...Vn_heads |
        #             |_              |               |              _|
        #                                              _          _
        #                                        ^    |            |
        # the dim of Q/K/V for a single head  seq len |< head dim >|
        #                                        v    |_          _|
        
        # split into Q, K, V
        # the split dimension is the last one, output shape for all q,k,v 
        # tensors will be [batch, seq_len, n_heads * head_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to -> [batch, seq_len, n_heads, head_dim] (separate heads)
        # then transpose to -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # output shape: [batch, n_heads, seq_len, head_dim]
        # basically the same as softmax((q @ k.mT)/sqrt(head_dim)) @ v
        # but more efficient 
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Transpose back to -> [batch, seq_len, n_heads, head_dim]
        # Then reshape to -> [batch, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(output)
        

class MLP(torch.nn.Module):

    def __init__(self, model_size, mlp_size, n_h_layers):

        super().__init__()
        
        self.layers = []

        self.layers.append(torch.nn.Linear(in_features = model_size,
                                           out_features = mlp_size))

        self.layers.append(torch.nn.GELU())
        
        for _ in range(n_h_layers-1):

            # assuming that the mlp has dimension model_size
            self.layers.append(torch.nn.Linear(in_features = mlp_size,
                                               out_features = mlp_size))

            self.layers.append(torch.nn.GELU())

        self.layers.append(torch.nn.Linear(in_features = mlp_size,
                                           out_features = model_size))

        self.layers.append(torch.nn.GELU())

        self.layers = torch.nn.ModuleList(self.layers)


    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x
        
        
class BaseBlock(torch.nn.Module):

    def __init__(self, model_size, n_heads, head_dim, mlp_dim, mlp_n_hlayers):

        super().__init__()

        self.mha = MultiHeadAttention(model_size, n_heads, head_dim)

        self.mlp = MLP(model_size, mlp_dim, mlp_n_hlayers)

        self.layernorm1 = torch.nn.LayerNorm(model_size)
        self.layernorm2 = torch.nn.LayerNorm(model_size)


    def forward(self, x):
        
        x = self.mha(self.layernorm1(x)) + x

        x = self.mlp(self.layernorm2(x)) + x

        return x


class Embedding(torch.nn.Module):
    
    def __init__(self, input_len, model_size):
        
        super().__init__()

        #self.embedding_matrix = torch.nn.Parameter(torch.randn((input_len, model_size)))
        self.embedding_matrix = torch.nn.Linear(input_len, model_size, bias=False)
        
        self.pe = PositionalEncoding(model_size = model_size)

    def forward(self, x):
        
        #emb = x @ self.embedding_matrix
        emb = self.embedding_matrix(x)

        # apply positional encoding
        emb = self.pe(emb)

        return emb
                

class Transformer(torch.nn.Module):
    
    def __init__(self, input_size, model_size, output_size, n_blocks, n_heads, head_dim, mlp_dim, mlp_n_hlayers):
        
        super().__init__()

        self.layers = []

        self.layers.append(Embedding(input_size, model_size))

        for _ in range(n_blocks):
            self.layers.append(BaseBlock(model_size, n_heads, head_dim, mlp_dim, mlp_n_hlayers))    

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self,x):

        for layer in self.layers:
            x = layer(x)

        return x


class VisionTransformer(torch.nn.Module):
    
    def __init__(self, obs_size, patch_size, model_size, n_blocks, n_heads, head_dim, mlp_dim, mlp_n_hlayers, output_size, device):

        super().__init__()

        # obs_size must be unbatched here
        transformer_input_size = obs_size[0]*patch_size*patch_size

        self.transformer = Transformer(transformer_input_size, 
                                       model_size, 
                                       output_size,
                                       n_blocks,
                                       n_heads,
                                       head_dim,
                                       mlp_dim,
                                       mlp_n_hlayers).to(device)

        # final projection layer to action space
        self.output_layer = torch.nn.Linear(in_features = model_size, out_features = output_size).to(device)

        self.patch_size = patch_size
        self.device = device

    def forward(self, tensor3d):

        # fixed cls token, it is used as a global aggregator
        batch,channels,h,w = tensor3d.shape
        cls_token = torch.zeros(batch,1,channels*self.patch_size*self.patch_size).to(self.device)
        linearized_patches = extract_patches(tensor3d, self.patch_size)
        linearized_patches = torch.cat((cls_token,linearized_patches), dim=1)
        transformer_output = self.transformer(linearized_patches) 

        # return last output
        output_token_emb = transformer_output[:,0,:]

        return self.output_layer(output_token_emb)


if __name__ == "__main__":

    obs_size = (3,100,100)
    rgb_img = torch.rand((1,*obs_size))

    # len of the flattened patch

    model = VisionTransformer(obs_size, 
                              patch_size = 12,
                              model_size = 128, 
                              n_blocks = 3,
                              n_heads = 3,
                              head_dim = 32,
                              mlp_dim = 128,
                              mlp_n_hlayers = 2,
                              output_size = 512, 
                              device = torch.device("cpu"))

    total_encoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("encoder parameters: ", total_encoder_params)

    output = model(rgb_img)

    print("output shape:", output.shape)
    output.mean().backward()
