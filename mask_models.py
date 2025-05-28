import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from functools import partial
from itertools import repeat
import collections.abc

class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'v1', model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        # self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        # print(self.model)
        self.model.eval()
        self.model.to(self.device)
        self.p = 8 if "v1" in model_type else 14
        self.stride = self.model.patch_embed.proj.stride

        assert self.stride[0] == self.p and self.stride[1] == self.p, f"stride should be equal to patch size. got {self.stride} and {self.p}"

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self.n_reg_tokens = self.model.num_register_tokens if "v2_reg" in model_type else None

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if 'v1' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
            print('Loaded dino_vitb8')
        elif 'v2_reg' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
            print('Loaded dinov2_vitl14_reg')
        elif 'v2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
            # model = torch.hub.load('dinov2', 'dinov2_vitl14', source='local', pretrained=False)
            # model.load_state_dict(torch.load('/home/chamuditha/Desktop/Phase_1/Ours/dinov2_vitl14_pretrain.pth'))
            # print('Loaded dinov2_vitl14')
        
            # print(model)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    # @staticmethod
    # def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
    #     """
    #     Creates a method for position encoding interpolation.
    #     :param patch_size: patch size of the model.
    #     :param stride_hw: A tuple containing the new height and width stride respectively.
    #     :return: the interpolation method
    #     """
    #     def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    #         npatch = x.shape[1] - 1
    #         N = self.pos_embed.shape[1] - 1
    #         if npatch == N and w == h:
    #             return self.pos_embed
    #         class_pos_embed = self.pos_embed[:, 0]
    #         patch_pos_embed = self.pos_embed[:, 1:]
    #         dim = x.shape[-1]
    #         # compute number of tokens taking stride into account
    #         w0 = 1 + (w - patch_size) // stride_hw[1]
    #         h0 = 1 + (h - patch_size) // stride_hw[0]
    #         assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
    #                                         stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
    #         # we add a small number to avoid floating point error in the interpolation
    #         # see discussion at https://github.com/facebookresearch/dino/issues/8
    #         w0, h0 = w0 + 0.1, h0 + 0.1
    #         patch_pos_embed = nn.functional.interpolate(
    #             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
    #             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
    #             mode='bicubic',
    #             align_corners=False, recompute_scale_factor=False
    #         )
    #         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    #         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    #         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    #     return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        # model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """

        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def get_embed_dim(self) -> int:
        """
        get the embedding dimension of the model.
        """
        return self.model.embed_dim
    
    def get_patch_size(self) -> int:
        """
        get the patch size of the model.
        """
        return self.p

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'token',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        if facet == 'token' and self.n_reg_tokens is not None:
            x = x[:, :, self.n_reg_tokens:, :]
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps

class LinearClassifier(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
        """
        super(LinearClassifier, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Conv2d(embedding_size, num_classes, kernel_size=1)

    def forward(self, a, b):
        """
        Forward pass of the classifier.
        Args:
            x: Input tensor of shape (batch_size, embedding_size, h, w)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes, h, w)
        """
        # Apply the classifier layers to the input embeddings
        x = a - b
        # remove the cls token
        x = x[:, 1:, :]
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
        logits = self.classifier(x)
        return logits


class CNNClassifier(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
        """
        super(CNNClassifier, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):
        """
        Forward pass of the classifier.
        Args:
            x: Input tensor of shape (batch_size, embedding_size, h, w)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes, h, w)
        """
        # Apply the classifier layers to the input embeddings
        x = a - b
        # remove the cls token
        x = x[:, 1:, :]
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
        logits = self.classifier(x)
        return logits


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation block for recalibrating feature maps. """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class ResidualBlock(nn.Module):
    """ Residual block with dilated convolutions and Squeeze-and-Excitation (SE) blocks. """
    def __init__(self, in_channels, out_channels, dilation=1, reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels, reduction=reduction)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Shortcut to match dimensions

    def forward(self, x):
        residual = self.shortcut(x)  # Identity shortcut
        out = self.conv(x)
        out = self.se(out)  # Apply SE block
        out += residual  # Add the residual (shortcut) connection
        return F.mish(out)  # Apply Mish activation after summation

class CDClassifier(nn.Module):
    """ The improved classifier with Residual Blocks, SE blocks, and Dilated Convolutions. """
    def __init__(self, embedding_size, num_classes=2):
        super(CDClassifier, self).__init__()
        self.layer1 = ResidualBlock(embedding_size, 256, dilation=2)  # Using dilated convolutions
        self.layer2 = ResidualBlock(256, 128, dilation=4)  # Another residual block
        self.layer3 = nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_X(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
    
    def forward(self, x_q, x_k, x_v):
        # print(x_q.shape)
        B_q, N_q, C = x_q.shape
        B_k, N_k, C_k = x_k.shape
        B_v, N_v, C_v = x_v.shape
        assert B_q == B_k == B_v and C == C_k == C_v, "Batch size and embedding dimension must match"

        qkv_q = self.qkv(x_q).reshape(B_q, N_q, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv_q.unbind(0)

        qkv_k = self.qkv(x_k).reshape(B_k, N_k, 3, self.num_heads, C_k // self.num_heads).permute(2, 0, 3, 1, 4)
        _, k, _ = qkv_k.unbind(0)

        qkv_v = self.qkv(x_v).reshape(B_v, N_v, 3, self.num_heads, C_v // self.num_heads).permute(2, 0, 3, 1, 4)
        _, _, v = qkv_v.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_X2(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
    
    def forward(self, x_q, x_k, x_v):
        B_q, N_q, C = x_q.shape
        B_k, N_k, C_k = x_k.shape
        B_v, N_v, C_v = x_v.shape
        assert B_q == B_k == B_v and C == C_k == C_v, "Batch size and embedding dimension must match"

        qkv_q = self.qkv(x_q).reshape(B_q, N_q, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv_q.unbind(0)

        qkv_k = self.qkv(x_k).reshape(B_k, N_k, 3, self.num_heads, C_k // self.num_heads).permute(2, 0, 3, 1, 4)
        _, k, _ = qkv_k.unbind(0)

        qkv_v = self.qkv(x_v).reshape(B_v, N_v, 3, self.num_heads, C_v // self.num_heads).permute(2, 0, 3, 1, 4)
        _, _, v = qkv_v.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_s = (q @ v.transpose(-2, -1)) * self.scale
        attn_s = attn_s.softmax(dim=-1)
        attn_s = self.attn_drop(attn_s)

        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C) + (attn_s @ k).transpose(1, 2).reshape(B_q, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class Block_X(Block):
        
            def __init__(
                    self,
                    dim,
                    num_heads,
                    mlp_ratio=4.,
                    qkv_bias=False,
                    drop=0.,
                    attn_drop=0.,
                    init_values=None,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm
            ):
                super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                init_values=init_values, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
                self.attn = Attention_X(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    
            def forward(self, x_q, x_k, x_v):
                x = x_q + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1(x_v))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

class Block_X2(Block):
        
            def __init__(
                    self,
                    dim,
                    num_heads,
                    mlp_ratio=4.,
                    qkv_bias=False,
                    drop=0.,
                    attn_drop=0.,
                    init_values=None,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm
            ):
                super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                init_values=init_values, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
                self.attn = Attention_X2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    
            def forward(self, x_q, x_k, x_v):
                x = x_q + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1(x_v))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

class AttnClassifier_X_V1(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V1, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_1 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_3 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_4 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_1(x, a, b)
        x = self.attention_blocks_2(x)
        x = self.attention_blocks_3(x, a, b)
        x = self.attention_blocks_4(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.classifier(x)

        return logits 

class AttnClassifier_X_V2(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V2, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_1 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_3 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_4 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_1(x, a, b)
        x = self.attention_blocks_2(x)
        x = self.attention_blocks_3(x, b, a)
        x = self.attention_blocks_4(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.classifier(x)

        return logits 

class AttnClassifier_X_V3(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V3, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_X1 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X2 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X3 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X4 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_1 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_X1(x, a, b) + self.attention_blocks_X2(x, b, a)
        x = self.attention_blocks_X3(x, a, b) + self.attention_blocks_X4(x, b, a)
        x = self.attention_blocks_1(x)
        x = self.attention_blocks_2(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.classifier(x)

        return logits 

class AttnClassifier_X_V4(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V4, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_X1 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X2 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X3 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X4 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_1 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.seg = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_X1(x, a, b) + self.attention_blocks_X2(x, b, a)
        x = self.attention_blocks_1(x)
        x = self.attention_blocks_X3(x, a, b) + self.attention_blocks_X4(x, b, a)
        x = self.attention_blocks_2(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.seg(x)

        return logits 

class AttnClassifier_X_V5(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V5, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_1 = Block_X2(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_3 = Block_X2(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_4 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_1(x, a, b)
        x = self.attention_blocks_2(x)
        x = self.attention_blocks_3(x, a, b)
        x = self.attention_blocks_4(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.classifier(x)

        return logits 

class AttnClassifier_X_V6(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V6, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_1 = Block_X2(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_3 = Block_X2(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_4 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Another conv layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer for num_classes (2)
        )

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_1(x, a, b)
        x = self.attention_blocks_2(x)
        x = self.attention_blocks_3(x, b, a)
        x = self.attention_blocks_4(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.classifier(x)

        return logits 

class AttnClassifier_X_V7(nn.Module):
    def __init__(self, embedding_size, height, width, num_classes=2, num_heads=8, num_attention_blocks=4):
        """
        Args:
            embedding_size: The size of the ViT embedding (typically 384 or 768 for ViT models).
            height: The height of the embedding grid (h).
            width: The width of the embedding grid (w).
            num_classes: Number of output classes. Default is 2 for binary segmentation.
            num_heads: Number of attention heads for the multi-head attention blocks.
            num_attention_blocks: Number of multi-head attention blocks to add before the classifier.
        """
        super(AttnClassifier_X_V7, self).__init__()
        
        self.embedding_size = embedding_size
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.num_attention_blocks = num_attention_blocks

        # # Define a learnable CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Positional encoding (for simplicity, use a learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (height * width) + 1, embedding_size))  # +1 for cls token

        # Define LayerNorm and attention blocks
        self.attention_blocks_X1 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X2 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X3 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_X4 = Block_X(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_1 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)
        self.attention_blocks_2 = Block(dim=embedding_size, num_heads=num_heads, mlp_ratio=4.0)

        self.norm_post = nn.LayerNorm(embedding_size)

        # Define a simple conv layer to process embeddings
        self.seg =CDClassifier(self.embedding_size, num_classes)

    def forward(self, a, b):

        x = a - b

        x = x + self.positional_encoding

        x = self.attention_blocks_X1(x, a, b) + self.attention_blocks_X2(x, b, a)
        x = self.attention_blocks_1(x)
        x = self.attention_blocks_X3(x, a, b) + self.attention_blocks_X4(x, b, a)
        x = self.attention_blocks_2(x)

        # LayerNorm after the attention blocks
        x = self.norm_post(x)

        # Remove cls token (keep the grid embeddings)
        x = x[:, 1:]

        # Reshape back to (batch_size, embedding_size, h, w)
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
    
        # Apply the classifier layers to the input embeddings
        logits = self.seg(x)

        return logits 
