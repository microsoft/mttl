import torch
from torch import nn


class TaskPredictionPlugin(nn.Module):
    name = "task_pred"

    def __init__(self, model, factor=1., detach=True):
        super().__init__()

        for _, selector in model.model.get_selectors().items():
            break

        self.predictor = nn.Linear(selector.module_logits.size(1), selector.module_logits.size(0))
        self.loss_fn = nn.CrossEntropyLoss()
        self.detach = detach
        self.factor = factor

    def compute_loss(self, model, batch, **kwargs):
        # tasks have already been propagated
        num = 0.
        loss = 0.

        for _, selector in model.get_selectors().items():
            logits = torch.sigmoid(selector.module_logits[batch["task_ids"]])
            if self.detach:
                logits = logits.detach()
            logits = self.predictor(logits)
            loss += self.loss_fn(logits, batch["task_ids"])
            num += 1.
        return (loss / num)


class ModuleSparsityPlugin(nn.Module):
    name = "mod_l1"

    def __init__(self, model, factor=1.):
        super().__init__()
        self.factor = factor
        self.num_free_bits = 1.

    def compute_loss(self, model, batch, **kwargs):
        # tasks have already been propagated
        num = 0.
        loss = 0.
        for _, selector in model.get_selectors().items():
            probs = torch.sigmoid(selector.module_logits[batch["task_ids"]])
            loss += torch.nn.functional.relu(probs.abs().mean(-1) - (self.num_free_bits / probs.size(-1))).mean()
            num += 1.
        return (loss / num)


class OrthoRegularizationPlugin(nn.Module):
    name = "ortho_reg"

    @staticmethod
    def orthogonal_loss_fn(param):
        type, param = param
        if type == "a":
            if len(param.shape) == 4:
                s, k, d, r = param.shape
                normed = nn.functional.normalize(param, p=2, dim=2)
                cosine_sim = torch.einsum('s i d r, s j d r -> s i j r', normed, normed)
            else:
                s, k, d = param.shape
                normed = nn.functional.normalize(param, p=2, dim=2)
                if k == 1:
                    normed = normed.view(s, d)
                    cosine_sim = torch.einsum('i d, j d -> i j', normed, normed)
                    return (cosine_sim ** 2).sum() / (s ** 2) - (1 / s)
                else:
                    cosine_sim = torch.einsum('i k d, j k d -> k i j', normed, normed)
                    return (cosine_sim ** 2).sum() / (k * s ** 2) - (1 / s)
        elif type == "b":
            s, k, r, d = param.shape
            normed = nn.functional.normalize(param, p=2, dim=-1)
            cosine_sim = torch.einsum('s i r d, s j r d -> s i j r', normed, normed)
        return (cosine_sim ** 2).sum() / (s * r * k ** 2) - (1 / k)

    def __init__(self, model, factor=1.):
        super().__init__()

        self.params = []
        for _, adapter in model.model.get_adapters().items():
            if hasattr(adapter, 'lora_a'):
                self.params.append(("a", adapter.lora_a))
            if hasattr(adapter, 'lora_b'):
                self.params.append(("b", adapter.lora_b))

        self.loss_fn = nn.CrossEntropyLoss()
        self.factor = factor

    def compute_loss(self, *args, **kwargs):
        loss = 0.
        for param in self.params:
            loss += self.orthogonal_loss_fn(param)
        return (loss / len(self.params))


class UnlikelihoodPlugin(nn.Module):
    name = "unlike"

    def __init__(self, model, factor=1.):
        super().__init__()

        self.n_tasks = model.hparams.n_tasks
        self.pad_token_id = model.pad_token_id
        self.factor = factor

    def compute_loss(self, model, batch, **kwargs):
        from utils import label_smoothed_nll_loss
        import torch.nn.functional as F

        task_ids = batch["task_ids"][:4]
        shuf_task_ids = torch.randint(0, self.n_tasks, task_ids.size(), device=task_ids.device)

        input_ids, target_ids = (
            batch["input_ids"][:4],
            batch["target_ids"][:4],
        )
        model.task_id_container["task_id"] = shuf_task_ids

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_ids)
        shuf_outputs = model.forward(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
            decoder_attention_mask=torch.ones_like(decoder_input_ids).float(),
        )
        loss, _ = label_smoothed_nll_loss(
            F.log_softmax(shuf_outputs.logits, dim=-1),
            target_ids,
            epsilon=0.,
            ignore_index=self.pad_token_id,
            reduction='none'
        )

        target_mask = (target_ids != self.pad_token_id).float()
        task_mask = (shuf_task_ids != task_ids).float()
        target_mask = target_mask * task_mask.unsqueeze(1)

        unlikely_loss = torch.log(torch.exp(-loss) + 1e-2) * target_mask
        unlikely_loss = unlikely_loss.sum() / (task_mask.sum() + 1e-2)
        return unlikely_loss


class MutualInformationPlugin(nn.Module):
    name = "mi_loss"

    def __init__(self, model, factor=1.):
        super().__init__()

        self.factor = factor

    def compute_loss(self, model, *args, **kwargs):
        num = 0.
        mi_loss = 0.
        for _, selector in model.get_selectors().items():
            probs = torch.sigmoid(selector.module_logits)
            # compute entropy of probs
            entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).sum(1)
            # compute marginal entropy of probs along first dimension
            marginal_entropy = -(probs.mean(0) * torch.log(probs.mean(0)) + (1 - probs.mean(0)) * torch.log(1 - probs.mean(0))).sum()
            # maximize mutual information
            mi_loss += entropy.mean() - marginal_entropy
            num += 1.
        return (mi_loss / num)
