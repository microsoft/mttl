import torch
from torch import nn
from torch.nn import functional as F

from mttl.models.poly import Selector
from mttl.cluster_tuning.cluster_reader import ClusterResult


class ClusterSelector(Selector):
    def __init__(self, config, soft=False):
        super().__init__()

        self.soft = soft
        self.n_skills = config.n_skills
        self.cluster_result = ClusterResult(config.example_to_ids_path)
        self.temperature = config.poly_selector_cluster_temp

        # just to get the device and the working dtype
        self.dummy_parameter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, routing_infos):
        # this should return a bs x n_clusters tensor that sums to 1
        if self.cluster_result.infos.input_type == "input":
            hashes = routing_infos.hashes
        else:
            hashes = routing_infos.instruction_hashes

        if self.soft:
            distances = self.cluster_result.get_distances_batch(hashes)
            distances = torch.tensor(
                distances,
                device=self.dummy_parameter.device,
            )
            routing = F.softmax(-distances / self.temperature, dim=-1).unsqueeze(1)
        else:
            cluster_ids = torch.tensor(
                [self.cluster_result.get_cluster(h) for h in hashes],
                device=self.dummy_parameter.device,
            )
            routing = F.one_hot(cluster_ids, num_classes=self.n_skills).unsqueeze(1)
        return routing.to(dtype=self.dummy_parameter.dtype)