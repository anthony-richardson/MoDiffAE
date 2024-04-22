import torch.nn as nn
import torch
from utils import dist_util


class SemanticRegressor(nn.Module):
    def __init__(self, modiffae_latent_dim, attribute_dim, semantic_encoder, cond_mean, cond_std):
        super(SemanticRegressor, self).__init__()
        self.semantic_encoder = semantic_encoder

        self.regressor = torch.nn.Linear(modiffae_latent_dim, attribute_dim)
        self.cond_mean = cond_mean
        self.cond_std = cond_std

    def forward(self, x):
        with torch.no_grad():
            emb = self.semantic_encoder(x)
        emb = self.normalize(emb)
        out = self.regressor(emb)
        return out

    def state_dict(self, *args, **kwargs):
        # Don't save the base model. Only store the regressor parameters.
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('semantic_encoder.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        # Never use strict loading for the semantic regressor.
        # We did not save the base model.
        return super().load_state_dict(state_dict, strict=False)

    def normalize(self, cond):
        cond = (cond - self.cond_mean.to(dist_util.dev())) / self.cond_std.to(
            dist_util.dev())
        return cond

    def denormalize(self, cond):
        cond = (cond * self.cond_std.to(dist_util.dev())) + self.cond_mean.to(
            dist_util.dev())
        return cond
