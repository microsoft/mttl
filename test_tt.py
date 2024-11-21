import torch
import torch.nn as nn
from transformers import BertConfig

class TensorPoly(nn.Module):
    def __init__(self, config):
        super().__init__()
        tensor_weights = [nn.Parameter(torch.rand(config.tensor_dim,1,config.tensor_dim,config.n_experts))]\
                    + [nn.Parameter(torch.rand(config.tensor_dim,config.n_experts,config.tensor_dim,config.n_experts))]*(config.tensor_count-2)\
                    + [nn.Parameter(torch.rand(config.tensor_dim,config.n_experts,config.tensor_dim,config.intermediate_size_multiple))]
        self.tensor_weights = nn.ParameterList(tensor_weights)


        self.pre_trained_layer = nn.Linear(config.hidden_size, config.intermediate_size)

        einsum_in = []
        einsum_out = ['','']
        routing_in = []
        
        in_current= 'a'
        out_current = 'i'
        r_begin = 'u'
        r_current = r_begin
        for _ in range(len(tensor_weights)):
            einsum_out[0] = einsum_out[0] + in_current
            einsum_out[1] = einsum_out[1] + out_current
            r_next = chr(ord(r_current) + 1)
            einsum_in.append(f'{in_current}{r_current}{out_current}{r_next}')
            in_current =  chr(ord(in_current) + 1)
            out_current = chr(ord(out_current) + 1)
            r_current = r_next
            routing_in.append(r_current)

        self.routing_tensor = nn.Parameter(torch.rand(config.n_tasks, config.tensor_count-1, config.n_experts))
        einsum_in_expr = ','.join(einsum_in+routing_in[:-1])
        einsum_out_expr= r_begin + einsum_out[0] + einsum_out[1] + r_current
        self.einsum_expr = f'{einsum_in_expr}->{einsum_out_expr}'
        # [a, b, c, d] -> in-dims
        # [i, j, k, l] -> out-dims
        # [u, v, w, x, y] -> bond-dims
        # self.einsum_expr = 'auiv,bvjw,cwkx,dxly,v,w,x->uabcdijkly'
        # v,w,x are routing weights for experts

    def forward(self, x, task_id):
        # We need to determine different routing

        print(*self.tensor_weights)
        print(*self.routing_tensor[task_id])
        breakpoint()
        tensor_weight = torch.einsum(self.einsum_expr, *self.tensor_weights, *self.routing_tensor[task_id].softmax(dim=-1)).flatten(start_dim=0, end_dim=len(self.tensor_weights)).flatten(start_dim=1, end_dim=-1)
        output = self.pre_trained_layer(x) + torch.matmul(x, tensor_weight)


        return output

if __name__ == '__main__':
    config = BertConfig()
    config.hidden_size = 256
    config.intermediate_size_multiple = 3
    config.intermediate_size = config.hidden_size * config.intermediate_size_multiple
    config.n_tasks = 3
    config.n_experts = 5
    config.tensor_dim = 4
    config.tensor_count = 4
    layer = TensorPoly(config)
    x = torch.rand(16, 128, config.hidden_size)
    print(layer(x, 1).shape)