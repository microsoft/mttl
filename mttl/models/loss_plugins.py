import torch
from torch import nn


class TaskPredictionPlugin(nn.Module):
    name = "task_pred"

    def __init__(self, model, factor=1.0, detach=True):
        super().__init__()

        for _, selector in model.model.get_selectors().items():
            break

        self.predictor = nn.Linear(
            selector.module_logits.size(1), selector.module_logits.size(0)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.detach = detach
        self.factor = factor

    def compute_loss(self, model, batch, **kwargs):
        # tasks have already been propagated
        num = 0.0
        loss = 0.0

        for _, selector in model.get_selectors().items():
            logits = torch.sigmoid(selector.module_logits[batch["task_ids"]])
            if self.detach:
                logits = logits.detach()
            logits = self.predictor(logits)
            loss += self.loss_fn(logits, batch["task_ids"])
            num += 1.0
        return loss / num


class ModuleSparsityPlugin(nn.Module):
    name = "mod_l1"

    def __init__(self, model, factor=1.0):
        super().__init__()
        self.factor = factor
        self.num_free_bits = 1.0

    def compute_loss(self, model, batch, **kwargs):
        # tasks have already been propagated
        num = 0.0
        loss = 0.0
        for _, selector in model.get_selectors().items():
            probs = torch.sigmoid(selector.module_logits[batch["task_ids"]])
            loss += torch.nn.functional.relu(
                probs.abs().mean(-1) - (self.num_free_bits / probs.size(-1))
            ).mean()
            num += 1.0
        return loss / num


class OrthoRegularizationPlugin(nn.Module):
    name = "ortho_reg"

    @staticmethod
    def orthogonal_loss_fn(param):
        type, param = param
        if type == "a":
            if len(param.shape) == 4:
                s, k, d, r = param.shape
                normed = nn.functional.normalize(param, p=2, dim=2)
                cosine_sim = torch.einsum("s i d r, s j d r -> s i j r", normed, normed)
            else:
                s, k, d = param.shape
                normed = nn.functional.normalize(param, p=2, dim=2)
                if k == 1:
                    normed = normed.view(s, d)
                    cosine_sim = torch.einsum("i d, j d -> i j", normed, normed)
                    return (cosine_sim**2).sum() / (s**2) - (1 / s)
                else:
                    cosine_sim = torch.einsum("i k d, j k d -> k i j", normed, normed)
                    return (cosine_sim**2).sum() / (k * s**2) - (1 / s)
        elif type == "b":
            s, k, r, d = param.shape
            normed = nn.functional.normalize(param, p=2, dim=-1)
            cosine_sim = torch.einsum("s i r d, s j r d -> s i j r", normed, normed)
        return (cosine_sim**2).sum() / (s * r * k**2) - (1 / k)

    def __init__(self, model, factor=1.0):
        super().__init__()

        self.params = []
        for _, adapter in model.model.get_adapters().items():
            if hasattr(adapter, "lora_a"):
                self.params.append(("a", adapter.lora_a))
            if hasattr(adapter, "lora_b"):
                self.params.append(("b", adapter.lora_b))

        self.loss_fn = nn.CrossEntropyLoss()
        self.factor = factor

    def compute_loss(self, *args, **kwargs):
        loss = 0.0
        for param in self.params:
            loss += self.orthogonal_loss_fn(param)
        return loss / len(self.params)
