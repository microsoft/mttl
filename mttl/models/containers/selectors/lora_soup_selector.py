from mttl.models.containers.selectors.base import (
    ExpertsAndWeightsSelectorOutput,
    forward_with_cache,
    SelectorConfig,
    Selector,
)
from mttl.models.containers.selectors.selector_output import (
    ALL_EXPERTS,
    BatchSequenceExpertsAndWeightsSelectorOutput,
)
from dataclasses import dataclass
import torch.nn as nn
import torch
from mttl.models.containers.selectors.lora_merge_eign_input_analysis import (
    AdaptiveLoRAMerger,
)
import torch.nn.functional as F
from mttl.models.library.expert import ExpertInfo



@dataclass
class LoraSoupSelectorConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_router", LoraSoupSelectorConfig)
class LoraSoupSelector(Selector):
    """
    LoraSoupSelector is a selector that uses a learnable routing mechanism to select experts.
    refer to the code: https://github.com/aksh555/LoRA-Soups
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.softmax = nn.Softmax(dim=-1)
        self.module_logits_dict = nn.ParameterDict()

    def _get_weights(self):
        wts_before_softmax = torch.cat(list(self.module_logits_dict.values()), dim=0)
        wts_after_softmax = self.softmax(wts_before_softmax)
        return wts_after_softmax

    def on_add_expert(self, expert_name, expert_info=None, is_default=False):
        if expert_name not in self.module_logits_dict:
            self.module_logits_dict[expert_name] = torch.nn.Parameter(
                torch.randn(1).to(self.device)
            )

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        # Get base weights from learned routing
        weights = self._get_weights()
        # aug_losses = self.info_container.routing_infos.aux_losses
        # aug_losses[self.layer_name] = self.aug_loss
        # Get expert adapters
        experts = list(self.module_logits_dict.keys())
        # print(self.layer_name, proj_coeffs)
        return ExpertsAndWeightsSelectorOutput(experts, weights)


@dataclass
class LoraSoupSelectorEignConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_router_eign", LoraSoupSelectorEignConfig)
class LoraSoupSelectorEign(LoraSoupSelector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.merger = None
        self.merger_cache = {}

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        if self.layer_name in self.merger_cache:
            # print(f"Using cached merger for {self.layer_name}")
            self.merger = self.merger_cache[self.layer_name]
        else:
            # print(f"Computing merger for {self.layer_name}")
            lora_a = kwargs["container"].lora_a
            lora_b = kwargs["container"].lora_b
            # Compute different coefficient methods
            math_lora_A = lora_a["zhan1993_mathqa_trained_from_lorasoup"]
            math_lora_B = lora_b["zhan1993_mathqa_trained_from_lorasoup"]
            code_lora_A = lora_a["zhan1993_code_trained_from_lorasoup"]
            code_lora_B = lora_b["zhan1993_code_trained_from_lorasoup"]
            self.merger = AdaptiveLoRAMerger(
                math_lora_A.T, math_lora_B.T, code_lora_A.T, code_lora_B.T
            )
            self.merger_cache[self.layer_name] = self.merger
        proj_coeffs = self.merger.compute_projection_based_coefficients(input, top_k=1)
        weights = torch.tensor(proj_coeffs).to(self.device)
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.u_plus = nn.Linear(hidden_dim, z_dim)
        self.delta_plus = nn.Linear(hidden_dim, z_dim)
        self.u_minus = nn.Linear(hidden_dim, z_dim)
        self.delta_minus = nn.Linear(hidden_dim, z_dim)

    def forward(self, R):
        h = self.net(R)
        return self.u_plus(h), self.delta_plus(h), self.u_minus(h), self.delta_minus(h)


@dataclass
class VariationalLoRSelectorConfig(SelectorConfig):
    encoder_hidden_dim: int = 256
    encoder_latent_dim: int = 64

    top_k: int = -1
    rkhs_dim: int = 512
    emb_dim: int = 128


@Selector.register("variational_lora_router", VariationalLoRSelectorConfig)
class VariationalLoRASelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        layer = kwargs["layer"]
        self.output_dim, self.input_dim = layer.out_features, layer.in_features

        self.hidden_dim = self.config.encoder_hidden_dim
        self.latent_dim = self.config.encoder_latent_dim
        # Encoder network for inferring latent variable distribution
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        # moe adapter selector
        self.top_k = self.config.top_k
        self.rkhs_dim = self.config.rkhs_dim
        self.emb_dim = self.config.emb_dim

        device = kwargs["layer"].weight.device
        # an expert level selector for each expert
        self.rkhs_exp = nn.Linear(self.emb_dim, self.rkhs_dim, device=device)
        self.rkhs_hid = nn.Linear(self.latent_dim, self.rkhs_dim, device=device)
        self.rkhs_embeddings = nn.Parameter(
            torch.empty((0, self.emb_dim), device=device)
        )

    def _get_weights(self, input):
        input_view = input.view(-1, input.shape[-1])
        return self.rkhs_hid(input_view).reshape(input.shape[0], input.shape[1], -1)

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        # do routing business on fp32
        input = input.to(dtype=self.rkhs_exp.weight.dtype)

        # Encode to get latent distribution
        u_plus, delta_plus, u_minus, delta_minus = self.encoder(input)
        z_plus = u_plus + torch.randn_like(u_plus) * torch.exp(delta_plus)
        z_minus = u_minus + torch.randn_like(u_minus) * torch.exp(delta_minus)
        z = z_plus + z_minus
        u_z = u_plus + u_minus
        delta_z = delta_plus + delta_minus
        
        # Get adapter weights based on latent variable
        rkhs_enc = self._get_weights(z)
        rkhs_emb = self.rkhs_exp(self.rkhs_embeddings)
        router_logits = torch.matmul(rkhs_enc, rkhs_emb.T)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.top_k > 0:
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(input.dtype)
        else:
            # soft routing
            selected_experts = ALL_EXPERTS

        if self.info_container is not None:
            g = self.info_container.routing_gates
            g.append(router_logits)

        kl_zp = self.kl_divergence(u_plus, delta_plus, u_z, delta_z)
        kl_zm = self.kl_divergence(u_minus, delta_minus, u_z, delta_z)

        kl_loss = kl_zp + kl_zm

        # Store auxiliary losses
        if self.info_container is not None:
            aug_losses = self.info_container.routing_infos.aux_losses
            aug_losses[self.layer_name] = kl_loss

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            experts=selected_experts, weights=routing_weights
        )

    # -----------------------------

    # KL Divergence Between Two Gaussians
    # -----------------------------
    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        return (
            0.5
            * torch.sum(
                logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1, dim=1
            ).mean()
        )

    def compute_kl_divergence_gaussian(self, mu, log_var):
        """Compute KL divergence between Gaussian and standard normal distribution"""
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return torch.mean(kl)

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        # just initialize the expert embeddings
        self.rkhs_embeddings.data = torch.cat(
            [
                self.rkhs_embeddings.data,
                torch.zeros(
                    1, self.emb_dim, device=self.device
                ).uniform_(-0.02, 0.02),
            ],
            dim=0,
        )
        self.encoder.to(self.device)


@dataclass
class LoraSoupPriorSelectorConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_prior_router", LoraSoupPriorSelectorConfig)
class LoraSoupPriorSelector(LoraSoupSelector):
    """
    LoraSoupPriorSelector extends LoraSoupSelector to incorporate prior routing information.
    It blends the learned routing weights with prior routing weights using a learnable alpha parameter.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Learnable parameter to control blending between prior and current routing
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _get_weights(self, prior_weights=None):
        # Get base weights from parent class
        current_weights = super()._get_weights()

        if prior_weights is not None:
            # Blend prior and current weights using learnable alpha
            weights = self.alpha * prior_weights + (1 - self.alpha) * current_weights
        else:
            weights = current_weights

        return weights

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        g = self.info_container.routing_gates
        prior_weights = None
        if len(g) > 0:
            prior_weights = g[-1]
        weights = self._get_weights(prior_weights)
        g.append(weights)
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)
