from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F



def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class PatchEmbed(nn.Module):

    def __init__(self, img_size=(224, 224), path_size=(16, 16), in_channel=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.path_size = path_size
        self.grid_size = (img_size[0] // path_size[0], img_size[1] // path_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # layers
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=path_size, stride=path_size)
        self.norm_layer = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input img size{H} * {W} doesn't match model size({self.img_size[0]} * {self.img_size[1]})"

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_layer(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layers = nn.ModuleList([
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,drop=drop_ratio)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class ViT(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_channel=3, embed_dim=128, depth=12, num_head=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., act_layer=None, norm_layer=None):
        super(ViT, self).__init__()
        self.features = self.embed_dim = embed_dim
        self.num_token = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channel, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_token, embed_dim))
        self.pos_drop = nn.Dropout(attn_drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # Transformer Encoder
        self.blocks = nn.Sequential(*[
            EncoderBlock(dim=embed_dim,num_heads=num_head,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                         drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                         act_layer=act_layer,norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)



    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # return self.pre_logits(x[:, 0]) # extract class_token
        return x

    def size_recovery(self,x):
        height = self.img_size[0] // self.patch_size[0]
        width = self.img_size[1] // self.patch_size[1]

        upsampled_output = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)

        return upsampled_output


    def forward(self, x):
        _, _, H, W = x.shape
        H = H // self.patch_size[1]
        W = W // self.patch_size[0]
        x = self.forward_features(x)
        B, N, C = x.shape
        x = x[:,1:,:] # take out cls_token
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.size_recovery(x)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    # Linear
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # Conv2d
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # LayerNorm
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_light_patch16_F(imgsize=(768, 1024), has_logits: bool = True):
    model = ViT(
        img_size=imgsize,
        patch_size=(128,128),
        embed_dim=128,
        depth=12,
        num_head=8,
        representation_size=768 if has_logits else None
    )
    return model


def vit_base_patch16_F(imgsize=(224,224), has_logits: bool = True):
    model = ViT(
        img_size=imgsize,
        patch_size=(16,16),
        embed_dim=128,
        depth=12,
        num_head=8,
        representation_size=768 if has_logits else None
    )
    return model

def vit_large_patch16_F(imgsize=(224,224), has_logits: bool = True):
    model = ViT(
        img_size=imgsize,
        patch_size=(16,16),
        embed_dim=768,
        depth=24,
        num_head=16,
        representation_size=1024 if has_logits else None
    )
    return model

def vit_huge_patch14_F(imgsize=(224,224), has_logits: bool = True):
    model = ViT(
        img_size=imgsize,
        patch_size=(16,16),
        embed_dim=1280,
        depth=32,
        num_head=16,
        representation_size=1280 if has_logits else None
    )
    return model



vit_light_patch16 = vit_light_patch16_F
vit_base_patch16 = vit_base_patch16_F
vit_large_patch16 = vit_large_patch16_F
vit_huge_patch14 = vit_huge_patch14_F





if __name__ == '__main__':
    visibel = torch.randn(2, 3, 480, 640)
    model = vit_base_patch16((480, 640))
    print(model(visibel).shape)
