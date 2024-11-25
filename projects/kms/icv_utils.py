# Taken from https://github.com/shengliu66/ICV

import torch
import torch.nn as nn
import torch.nn.functional as F


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


class ICVLayer(nn.Module):

    def __init__(self, icv, lam):
        super(ICVLayer, self).__init__()
        self.icv = icv.view(1, -1)
        self.norm_icv = F.normalize(icv.view(1, 1, -1), dim=-1)
        self.lam = lam

    def forward(self, x):
        if self.icv is not None:
            dtype = x.dtype
            x = x.float()
            original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            """
            directions_all = []
            y = 0
            for i in range(len(self.icv)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x, -self.icv[i][None,None,:], dim=-1)).unsqueeze(-1)
                y += self.lam * lambda_sim * F.normalize(self.icv[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.icv)
            y = self.lam * F.normalize(self.icv[0], dim=-1).repeat(1,x.shape[1],1) 
            x = F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm
            """
            # Add the ICV
            x = F.normalize(x, p=2, dim=-1) + self.lam * self.norm_icv
            # Renormalize
            x = F.normalize(x, p=2, dim=-1) * original_norm
            return x.to(dtype)
        else:
            return x
