from typing import Tuple, List, Union
import torch
from torch import nn
from torchvision import models
from torchvision.models import list_models as tv_lm
from torchvision import models
from timm import list_models as timm_lm
from timm import create_model


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class RegModel(nn.Module):
    def __init__(self, backbone='resnet18', **kwargs):
        super(RegModel, self).__init__()
        self.mm_conv = nn.Conv2d(1, 3, 1, 1, bias=False)
        
        if backbone in tv_lm():
            self.backbone: nn.Module = eval(f'models.{backbone}')(**kwargs)        
        else:
            self.backbone: nn.Module = create_model(backbone, **kwargs)
        
        last_layer = getattr(self.backbone, 'fc', None)
        
        if last_layer is not None:
            last_layer = getattr(self.backbone, 'head', None)
            self.backbone.head = nn.Linear(last_layer.in_features, 1)    
        else:
            self.backbone.fc = nn.Linear(last_layer.in_features, 1)
            
    def forward(self, x):
        x = self.mm_conv(x)
        x = self.backbone(x)
        return x
        

class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            input_image_size: Tuple[int, int],
            mask_in_chans: int,
            act: nn.GELU
    ):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.pe_layer = nn.PositionEmbedding(image_embedding_size)
        