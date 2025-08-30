from functools import partial
import timm.models.vision_transformer
import torch
import torch.nn as nn


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling, for age regression """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.head_mi = Prediction_head(embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # exclude cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        return self.head_mi(x)


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DinoV2WithHead(nn.Module):
    """ Wraps a DINOv2 model for age regression """
    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.head_mi = Prediction_head(embed_dim)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x[:, 1:, :].mean(dim=1)  # global pooling
        x = self.backbone.fc_norm(x)
        return self.head_mi(x)


def RETFound_dinov2(args, **kwargs):
    backbone = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    backbone.head_mi = Prediction_head(embed_dim=1024)

    def forward(x):
        x = backbone.forward_features(x)
        x = x[:, 1:, :].mean(dim=1)
        x = backbone.fc_norm(x)
        return backbone.head_mi(x)

    backbone.forward = forward
    return backbone


class Prediction_head(nn.Module):
    """ Simple regression head: Linear -> ReLU -> Dropout -> Linear(1) """
    def __init__(self, embed_dim):
        super().__init__()
        self.double_linear = nn.Sequential(
            nn.Linear(embed_dim, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        return self.double_linear(x)
