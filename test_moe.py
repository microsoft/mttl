import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Output probabilities for expert selection
        return F.softmax(self.fc(x), dim=-1)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Initialize experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

        # Initialize gating network
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # Get gating weights
        gate_outputs = self.gating_network(x)  # Shape: [batch_size, num_experts]

        # Select top-k experts
        topk_values, topk_indices = torch.topk(
            gate_outputs, self.top_k, dim=-1
        )  # Shape: [batch_size, top_k]

        # Normalize top-k weights
        topk_weights = F.softmax(topk_values, dim=-1)  # Shape: [batch_size, top_k]

        # Initialize output
        output = torch.zeros(
            x.size(0), self.experts[0].fc2.out_features, device=x.device
        )

        # Aggregate outputs from top-k experts
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_weights[:, i]

            # Gather expert outputs
            expert_output = torch.stack(
                [
                    self.experts[idx](x[j].unsqueeze(0))
                    for j, idx in enumerate(expert_idx)
                ]
            )

            # Weight expert outputs and sum
            output += expert_weight.unsqueeze(1) * expert_output

        return output


# Example Usage
if __name__ == "__main__":
    batch_size = 4
    input_dim = 16
    hidden_dim = 32
    output_dim = 16
    num_experts = 5
    top_k = 2

    # Input token embeddings (batch of tokens)
    token_embeddings = torch.randn(batch_size, input_dim)

    # Initialize MoE model
    moe_model = MixtureOfExperts(input_dim, hidden_dim, output_dim, num_experts, top_k)

    # Forward pass
    output = moe_model(token_embeddings)

    print("Output Shape:", output.shape)  # Should be [batch_size, output_dim]
