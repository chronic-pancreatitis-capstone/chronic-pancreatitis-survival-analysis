import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Transformer Parameters:
EMBED_DIM    = 256
META_DIM     = 2       
NUM_LAYERS   = 2
NUM_HEADS    = 4
DROPOUT      = 0.1
MODE         = "COX"

# ============================================================================
# 1. Transformer aggregator
# ============================================================================

class TransformerAggregator(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        meta_dim=META_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, meta, mask):
        """
        x:    (B, L, D)
        meta: (B, L, meta_dim)
        mask: (B, L)  where True=valid, False=pad
        """
        B, L, D = x.shape
        assert meta.shape[:2] == (B, L)

        # Simple meta → positional embedding
        pos = self.meta_mlp(meta)   # (B,L,D)
        x = x + pos

        # Transformer expects src_key_padding_mask with True for PAD positions
        src_key_padding_mask = ~mask.bool()  # True=PAD
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling over slices
        mask_float = mask.unsqueeze(-1).float()
        x_sum = (x * mask_float).sum(dim=1)
        lengths = mask_float.sum(dim=1).clamp(min=1)
        patient_vec = x_sum / lengths

        return self.out_proj(patient_vec)
    

# ============================================================================
# 2. Cox Model
# ============================================================================

import torch.nn as nn

class CoxRiskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

class DeepSurvHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)




class CoxModel(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        meta_dim=META_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        encoder=None,
        mode = MODE,
        embedMode = False
    ):
        super().__init__()
        self.encoder = encoder
        assert self.encoder is not None

        # Freeze SAM-Med2D encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # default eval

        self.aggregator = TransformerAggregator(
            embed_dim=embed_dim,
            meta_dim=meta_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.risk_head = DeepSurvHead(in_dim=embed_dim) if mode == "DEEP" else CoxRiskHead(in_dim=embed_dim)
        self.embedMode = embedMode
        
    def _ensure_encoder_eval(self):
        """
        Keep encoder in eval mode even when the overall model is set to train().
        This avoids dropout / BN randomness in the frozen backbone.
        """
        if self.encoder.training:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep encoder in eval mode
        self._ensure_encoder_eval()
        return self

    def eval(self):
        super().eval()
        self._ensure_encoder_eval()
        return self

    def forward(self, imgs, meta, mask):
        """
        imgs: (B, L, 3, 256, 256)
        meta: (B, L, 2)
        mask: (B, L)
        """
        
        self._ensure_encoder_eval()

        device = imgs.device
        B, L, C, H, W = imgs.shape

        # Flatten all slices
        imgs_flat = imgs.view(B * L, C, H, W)

        # SAM-Med2D encoder (frozen, no grad)
        with torch.no_grad():
            feat = self.encoder(imgs_flat)          # (B*L, 256, h, w)
            feat = feat.mean(dim=[2, 3])           # (B*L, 256)

        feat = feat.view(B, L, -1).float()         # (B,L,D)

        patient_vec = self.aggregator(
            feat,
            meta.to(device),
            mask.to(device),
        )
        if self.embedMode == False:
            risk = self.risk_head(patient_vec)         # (B,)
            return risk
        else:
            return patient_vec



def cox_ph_loss(risk, time, event):
    """
    Standard Cox partial log-likelihood (negative).
    Sort times in DESCENDING order so that logcumsumexp(risk) gives the
    risk set for each individual.
    """
    order = torch.argsort(time, descending=True)
    time  = time[order]
    event = event[order]
    risk  = risk[order]

    hazard = torch.logcumsumexp(risk, dim=0)
    loglik = (risk - hazard) * event
    loss = -loglik.sum() / (event.sum() + 1e-8)
    return loss
    

