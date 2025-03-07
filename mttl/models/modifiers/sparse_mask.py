from dataclasses import dataclass
import os
import re
from typing import List, Union
import numpy as np
import torch
from torch import nn
import math
import os
from torch import nn
import torch
import math
import bitsandbytes as bnb
import types
import copy
from mttl.utils import logger
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import (
    MergeableAdapter,
    ModifyMixin,
    ModifierConfig,
    Adapter,
)
import random
import numpy as np
def load_mask(f_name):
    destination_type, _  = f_name.split('://')
    if destination_type=='hf':
        from huggingface_hub import hf_hub_download
        destination_type, f_name = f_name.split('://')
        repo_id=('/').join(f_name.split('/')[:2])
        task_name = f_name.split('/')[-1]
        f_path=hf_hub_download(repo_id=repo_id, filename=task_name)
        mask_dict = np.load(f_path, allow_pickle=True)['arr']
    elif destination_type=='local':
        mask_dict = np.load(f'{f_name}.npz', allow_pickle=True)['arr']
    return mask_dict

def save_mask(module, f_name):
    """
    to load the saved mask use the `load_mask` function
    """
    mask_dict = {}
    import numpy as np
    for m_name, m in dict(module.named_modules()).items():
        if 'sparse_layer' in m_name:
            mask_dict[m_name] = torch.nonzero(m.weight_mask.data).cpu().numpy()
    destination_type=f_name.split('://')[0]
    # save in local dir
    if destination_type=='local':
        destination_type, f_name = f_name.split('://')
        np.savez_compressed(f'./{f_name}.npz', arr=mask_dict)

    # upload to hf 
    elif destination_type=='hf':
        from huggingface_hub.hf_api import upload_file as hf_upload_file
        destination_type, f_name = f_name.split('://')
        repo_id=('/').join(f_name.split('/')[:2])
        task_name = f_name.split('/')[-1]
        path_in_repo=f'{task_name}.npz'
        os.makedirs('./temp/test_library/', exist_ok=True)
        local_file_path = f'./temp/test_library/{path_in_repo}'
        np.savez_compressed(local_file_path, arr=mask_dict)
        
        hf_upload_file(path_or_fileobj=local_file_path, # path saved in local machine
                      path_in_repo=path_in_repo,     # path with in repo
                      repo_id=repo_id)            #
     # exact local dir is provided
    else:
        np.savez_compressed(f'./{f_name}.npz', arr=mask_dict)

def update_noise_space(module, grad_score, accepted_score, noise_params_to_keep):
    """
    m.nextk_idx is updated, which identifies the noise space used to sample noise in forward step
    """
    assert len(grad_score.shape)==1, "grad_score has to be 1D array to get accurate noise-index, 2D grad matrix need to be flatten() for proper noise-mask calculation"
    if module.noise_cat == 'targeted_noise':
        lower_bound, _ = torch.topk(grad_score, noise_params_to_keep, sorted=True)
        lower_bound = lower_bound[-1]
        noise_mask = ((grad_score<accepted_score).float()*(grad_score>=lower_bound).float())
    elif module.noise_cat == 'random_noise':
        noise_mask = (grad_score<accepted_score).float()

    if noise_mask.sum()==0:
        raise ValueError("noise mask is all 0, no noise is added. Set `activate_noise=False` if no noise is expected during training")    
    assert len(noise_mask.shape)==1, "noise-mask has to be 1D array to get accurate noise-index, check the inputs and noise_mask calculation"

    module.nextK_idx = torch.where(noise_mask==1)[0]


class MatrixBlockIndexer():
    """
    Example use case:
    M, N, BLOCK_SIZE = 4, 4, 2
    indexer = MatrixBlockIndexer(M, N, BLOCK_SIZE)
    # Example: Get indices for block 3
    block_i = 3
    indices = indexer.get_block_indices(block_i)
    indices_list = [indexer.get_block_indices(i) for i in range(indexer.L_BLOCK)]
    """
    def __init__(self,M=4,N=4,BLOCK_SIZE=2):

        if M % BLOCK_SIZE != 0 or N % BLOCK_SIZE != 0:
            raise ValueError("M  and must be divisible by BLOCK_SIZE")

        self.M = M
        self.N = N
        self.BLOCK_SIZE = BLOCK_SIZE

        self.calculate_params()
    
    def convert_mat_2_block(self, W_idx):
        # reshape it to get the indices of every block
        W_idx = W_idx.reshape(self.M // self.BLOCK_SIZE, self.BLOCK_SIZE, self.N // self.BLOCK_SIZE, self.BLOCK_SIZE)
        W_idx = W_idx.permute(0, 2, 1, 3).flatten(0, 1)
        return W_idx

    def calculate_params(self):
        # build a matrix of indices
        W_idx = torch.arange(self.M * self.N ).reshape(self.M, self.N)
        # instead of storing all the indices, store the indices of one block,
        W_idx = self.convert_mat_2_block(W_idx)
        # and store the offset for every block
        self.first_block_idx = W_idx[0]
        self.block_offset = W_idx[:, 0, 0].flatten()
        self.L_BLOCK = len(self.block_offset)

    def get_block_indices(self, block_i):
        block_i_indices = self.block_offset[block_i] + self.first_block_idx
        return block_i_indices
        #get_i = lambda i : self.block_offset[i] + self.first_block_idx




def get_block_mask(m):
    """
    BLOCK-SPARSE mask calculation
    """
    num_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask)*m.keep_ratio)
    num_blocks_to_keep = int(num_params_to_keep/(m.BlockwiseConvolution.BLOCK_SIZE**2))                    
    # get block scores
    block_grad = m.BlockwiseConvolution.convert_mat_2_block(m.sparse_layer.weight_mask.grad)
    block_score = block_grad.sum(dim=(1,2))

    # find the top-k blocks
    threshold, topk_block_idx = torch.topk(block_score, num_blocks_to_keep, sorted=True)
    accepted_score = threshold[-1]

    # get mask-indices of the top-k blocks
    keep_masks_idx = [m.BlockwiseConvolution.get_block_indices(i) for i in topk_block_idx]
    keep_masks_idx=torch.stack(keep_masks_idx).flatten().to(m.layer.weight.device)

    # get the mask
    keep_masks = torch.zeros_like(m.sparse_layer.weight)
    keep_masks.flatten().scatter_add_(0, keep_masks_idx, 
                                        torch.ones(keep_masks_idx.shape, dtype=torch.float32, device=m.layer.weight.device))
    if m.activate_noise:
        # update the noise space
        noise_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask)*(m.keep_ratio+m.noise_space_ratio))
        noise_blocks_to_keep = int(noise_params_to_keep/(m.BlockwiseConvolution.BLOCK_SIZE**2))
        update_noise_space(module=m,
                            grad_score=block_score, 
                            accepted_score=accepted_score, 
                            noise_params_to_keep=noise_blocks_to_keep)

    return keep_masks



def get_regular_sparse_mask(m):
    """
    parameter-wise sparse calculation
    """
    num_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask)*m.keep_ratio)
    threshold, _ = torch.topk(m.sparse_layer.weight_mask.grad.flatten(), num_params_to_keep, sorted=True)
    accepted_score = threshold[-1]
    keep_masks = (m.sparse_layer.weight_mask.grad>=accepted_score).float()

    if m.activate_noise:
        noise_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask)*(m.keep_ratio+m.noise_space_ratio))
        update_noise_space(module=m, 
                            grad_score=m.sparse_layer.weight_mask.grad.flatten(), 
                            accepted_score=accepted_score, 
                            noise_params_to_keep=noise_params_to_keep)
    return keep_masks





def make_sparse_model_during_training(module, batch, print_statement=False, parameter_selection_procedure='per_layer'): 
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule
    # (1) preprocess the sparse-layers
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.preprocess_for_mask_update()

    # (2) collect grads
    from mttl.models.utils import transfer_batch_to_device
    #batch = transfer_batch_to_device(batch, module.device)
    loss = module.forward(batch)
    loss.backward()


    assert parameter_selection_procedure in ['model','per_layer'], "choose the right `parameter_selection_procedure`"

    # (3) compute mask
    # (a) layer-wise
    if parameter_selection_procedure=='per_layer':
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                if m.sparse_cat == 'block_sparse':
                    keep_masks = get_block_mask(m)
                    ## --- sample noise-block-idx
                    # block_noise_idx = sample(m.nextK_idx)
                    # noise_masks_idx = [m.BlockwiseConvolution.get_block_indices(i) for i in block_noise_idx]
                    # noise_masks_idx=torch.stack(noise_masks_idx).flatten().to(m.layer.weight.device)
                    # print('check')
                elif m.sparse_cat == 'regular_sparse':
                    keep_masks = get_regular_sparse_mask(m)

                # (4) revert back to original state
                # (a) reverse the require-grad: Turn on for `weight` and turn-off for `weight_mask`
                # (b) convert `module` back to `cpu`
                m.revert_weight_grad_and_update_mask(keep_masks)
                #if save_mask_indx: mask_indx.append(torch.nonzero(keep_masks).data.cpu().numpy()) # nonzero finds the ind


    # (b) based on whole-net TODO
    # b.1 compute score
    elif parameter_selection_procedure=='model':
        num_params_to_keep = 0
        grads = []
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                assert m.sparse_cat == 'regular_sparse',  "parameter_selection_procedure over `model` is not implemented for `block_sparse`"
                num_params_to_keep += int(torch.numel(m.sparse_layer.weight_mask)*m.keep_ratio)
                grads.append(m.sparse_layer.weight_mask.grad.flatten().cpu())


        threshold, _ = torch.topk(torch.stack(grads).flatten(), num_params_to_keep, sorted=True)
        accepted_score = threshold[-1]
        # b.2 mask
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                keep_masks = (m.sparse_layer.weight_mask.grad>=accepted_score).float()
                if print_statement: print('sparsity', (keep_masks.sum()/m.sparse_layer.weight_mask.numel())*100, 'expected', m.keep_ratio*100)
                m.revert_weight_grad_and_update_mask(keep_masks)



def mod_forward(self, x):
    return torch.nn.functional.linear(x, self.weight*self.weight_mask, self.bias)

"""consider adding random noise"""
def mod_noisy_forward(self, x):
    #return torch.nn.functional.linear(x, self.weight*self.weight_mask, self.bias)
    # return torch.nn.functional.linear(x, self.weight*self.weight_mask+self.noise*self.noise_var, self.bias)
    return torch.nn.functional.linear(x, self.weight*self.weight_mask+self.noise, self.bias)
    #return torch.nn.functional.linear(x, self.weight*self.weight_mask+self.weight*self.weight_mask, self.bias)

@dataclass
class SparseMaskConfig(ModifierConfig):
    keep_ratio: float = 1.0
    noise_add_ratio: float = 1.0
    noise_space_ratio: float = 1.0
    activate_noise: bool = True
    mask_cat: str = 'scatter'
    noise_cat: str = 'targeted_noise' # 'targeted_noise' or 'random_noise'
    training_mode: bool = True
    BLOCK_SIZE: int= 16             # 16x 
    sparse_cat: str="block_sparse"  # ['block_sparse','regular_sparse']

@register_modifier("sparse_mask_adapter", config_cls=SparseMaskConfig)
class SparseMaskAdapter(ModifyMixin):
    def __init__(
        self,
        config: SparseMaskConfig,
        layer: nn.Module,
        **kwargs,
    ):
        super().__init__()

        self.layer=layer
        input_dim, output_dim = self.layer.in_features, self.layer.out_features
        self.param_shape = self.layer.weight.shape
        self.param_num = self.layer.weight.numel()

        self.sparse_cat = config.sparse_cat 
        assert self.sparse_cat in ['block_sparse','regular_sparse'], "Choose `sparse_cat` from ['block_sparse','regular_sparse'] "
        
        # weight initialization
        self.sparse_layer = nn.Linear(input_dim, output_dim).to(device=layer.weight.device)
        self.sparse_layer.weight = nn.Parameter(torch.zeros(self.sparse_layer.weight.shape))
        self.sparse_layer.bias = nn.Parameter(torch.zeros(self.sparse_layer.bias.shape))

        
        if self.sparse_cat == 'block_sparse':
            self.BLOCK_SIZE = config.BLOCK_SIZE
            self.BlockwiseConvolution = MatrixBlockIndexer(M=input_dim, N=output_dim, BLOCK_SIZE=self.BLOCK_SIZE)
   
        # mask initialization
        self.sparse_layer.weight_mask = torch.ones(self.sparse_layer.weight.shape).to(device=layer.weight.device)
        self.mask_cat= config.mask_cat
        self.keep_ratio = config.keep_ratio
        self.keed_mask_idx = None                 # will be initialized during training        

 

        self.training_mode = config.training_mode
        # noise initialization
        self.activate_noise = config.activate_noise    # if True, uses a noisy parameter training
        assert type(self.activate_noise) == bool

        # update forward step
        if self.training_mode == False:
            # forward function
            self.patch_forward()                
        else:
            # forward function        
            if self.activate_noise:
                # 'nextK_idx' will be initialized during training, used in adding noise
                # (a) if noise_cat=='targeted_noise': stores the idx of most % important weights after the trainable params
                # (b) if noise_cat=='random_noise': a larger-matrix keeps idx of everything except the trainable params
                self.nextK_idx = None
                self.noise_cat = config.noise_cat             
                assert self.noise_cat in ['targeted_noise','random_noise']

                self.noise_space_ratio = config.noise_space_ratio # % of parameter space that we sample noise idx from
                self.noise_add_ratio = config.noise_add_ratio  # % of parameter where noise will be added

                assert (self.noise_space_ratio > self.keep_ratio and self.noise_space_ratio > self.noise_add_ratio), "since noise indx is sampled from `noise_space_ratio`, it's value must be greater than `keep_ratio` and `noise_space_ratio` "
                
                # TODO: I don't think we need two matrix here
                #self.zero_matrix = torch.zeros(self.param_shape, device=self.layer.weight.device)
                self.sparse_layer.noise = torch.zeros(self.param_shape, device=self.layer.weight.device)
                self.noise_mean = None
                self.noise_std = None
                self.turn_off_require_grad()


                self.patch_noisy_forward()
            else:
                self.patch_forward()
        # used for noise calculcation
        self.sampled_indices = None 
        self.noise = None

    # forward without noise
    def patch_forward(self):
        self.sparse_layer.forward = types.MethodType(mod_forward, self.sparse_layer)

    # forward with noise
    def patch_noisy_forward(self):
        self.sparse_layer.forward = types.MethodType(mod_noisy_forward, self.sparse_layer)

    def turn_off_require_grad(self):
        self.sparse_layer.noise.requires_grad=False 

    @torch.no_grad()
    def convert_sparse_weight_to_1D(self):
        assert len(self.sparse_layer.weight.shape)==2, print('sparse_layer.weight is already converted to 1D')
        self.sparse_layer.weight = nn.Parameter(self.sparse_layer.weight.flatten()[self.keep_mask_idx].data).to(self.layer.weight.device)
        
    
    def calculate_noise(self, sampled_indices):
        if self.noise_mean == None:
            masked_w = self.sparse_layer.weight.flatten()[self.keep_mask_idx]
            noise_mean = masked_w.mean()
            noise_std = masked_w.std()
            noise = torch.normal(mean=noise_mean, 
                        std=noise_std, 
                        size=sampled_indices.shape, dtype=torch.float32, device=self.layer.weight.device)
        else:
            noise = torch.normal(mean=self.noise_mean, 
                        std=self.noise_std, 
                        size=sampled_indices.shape, dtype=torch.float32, device=self.layer.weight.device)
        return noise
    

    
    def sample_noise_idx(self):

        if self.sparse_cat == 'regular_sparse':
            # For both `random_noise` and `targeted_noise` we sample indices
            # Determine the number of elements to sample (20%)
            num_samples = int(self.param_num*self.noise_add_ratio)   # number of sampled-parameters
            # number of max-parameters
            num_elements = len(self.nextK_idx)
            # Randomly sample indices
            sampled_indices = self.nextK_idx[torch.randint(0, num_elements, (num_samples,)) ]

        elif self.sparse_cat == 'block_sparse':
            # For both `random_noise` and `targeted_noise` we sample indices
            # Determine the number of elements to sample (20%)
            total_num_blocks = self.param_num/(self.BLOCK_SIZE**2)
            num_samples = int(total_num_blocks*self.noise_add_ratio)   # number of sampled-parameters
            # number of max-parameters
            num_elements = len(self.nextK_idx)
            # Randomly sample indices
            sampled_indices = self.nextK_idx[torch.randint(0, num_elements, (num_samples,)) ]

            # For block-sparse the indices and block-indices, that we need to convert to parameter positions
            sampled_indices = [self.BlockwiseConvolution.get_block_indices(i) for i in sampled_indices]
            sampled_indices=torch.stack(sampled_indices).flatten().to(self.layer.weight.device)


        return sampled_indices

    @torch.no_grad()
    def generate_noise(self):
        if self.nextK_idx!=None:
            """STEP 1: Generate a random one-hot noise matrix that consitutes 20% of the nextK_idx important weights"""
            
            if (self.sampled_indices==None or self.noise==None): 
                self.sampled_indices = self.sample_noise_idx()  
                self.noise = self.calculate_noise(self.sampled_indices)
            else:
                if random.randint(0,1)==0:
                    self.sampled_indices = self.sample_noise_idx()
                else:
                    self.noise = self.calculate_noise(self.sampled_indices)


            # option 1: problem need to create torch matrix in GPU in every gradient step
            self.sparse_layer.noise = torch.zeros(self.param_shape, device=self.layer.weight.device)
            #self.sparse_layer.noise.flatten().index_add_(0, sampled_indices, noise)
            self.sparse_layer.noise.flatten().scatter_add_(0, self.sampled_indices, self.noise)

            # option 2: problem need to keep track of two matrix in memory and scatter(-noise) leave residual value
            # if self.zero_matrix.device != self.layer.weight.device:
            #     self.zero_matrix = self.zero_matrix.to(self.layer.weight.device)
            # self.zero_matrix.flatten().scatter_add_(0, sampled_indices, noise)
            # # add noise
            # self.sparse_layer.noise = self.zero_matrix.clone()
            # # reset to zero
            # self.zero_matrix.flatten().scatter_add_(0, sampled_indices, -noise)  # there is a vary small value 1e-8

        else:
            # Ones matrix, non_trainable as default. sync the device
            if self.sparse_layer.noise.device != self.sparse_layer.weight.device:
                self.sparse_layer.noise = self.sparse_layer.noise.clone().to(self.sparse_layer.weight.device)
        self.turn_off_require_grad()



    def data_preprocess(self,x):
        sparse_model_dtype = self.sparse_layer.weight.dtype
        return x.to(sparse_model_dtype)
    
    def confirm_sync_device(self):
        with torch.no_grad():
            # sync the device of the `weight_mask` with module layers
            if self.sparse_layer.weight_mask.requires_grad:
                # when gradient graph is true
                if self.sparse_layer.weight.device!=self.layer.weight.device:
                    self.sparse_layer.weight_mask = self.sparse_layer.weight_mask.to(self.layer.weight.device)
            else:
                # otherwise
                # required `.clone()` before device transfer, otherwise getting following error
                # """RuntimeError: Inference tensors cannot be saved for backward. 
                #    To work around you can make a clone to get a normal tensor and use it in autograd."""
                #if self.sparse_layer.weight_mask.device!=self.layer.weight.device:
                self.sparse_layer.weight_mask = self.sparse_layer.weight_mask.clone().to(self.layer.weight.device)
                #weight_mask = self.sparse_layer.weight_mask.to(self.layer.weight.device)
                #self.sparse_layer.weight_mask = weight_mask
        
    
    def forward(self, input):
        # `weight_mask` requires to sync with `weight` device, as default only looks module 
        
        self.confirm_sync_device()  # TODO, need to remove this, it should only require once before training        
        output = self.layer(input)

        # TODO
        # with generate noise will make self.noise no-zero matrix
        if (self.training_mode and self.activate_noise):
            self.generate_noise()
        #self.add_random_noise()
        #print(self.layer.weight.device, self.sparse_layer.weight.device, self.sparse_layer.weight_mask.device)

        if self.sparse_layer.weight.device!=self.sparse_layer.weight_mask.device:
            print(self.sparse_layer.weight.device, self.sparse_layer.weight_mask.device)
        try:
            sparse_output = self.sparse_layer(self.data_preprocess(input)) # Bfloat16-->Float32

        except:
            print(self.sparse_layer.weight.device, self.sparse_layer.weight_mask.device)
        return output + sparse_output.to(input.dtype)                  # Float32-->Bfloat16



    """
    - prepare the mask and corresponding weight for mask-gradient calculation step during mask update
    - in this step, we want to compute weight "only" w.r.t. `weight_mask`
    """
    def preprocess_for_mask_update(self):
        # Turn off the gradient for weight
        self.sparse_layer.weight.requires_grad = False
        # init the mask
        self.sparse_layer.weight_mask = nn.Parameter(torch.ones(self.sparse_layer.weight_mask.shape, 
                                                                device=self.layer.weight.device))
        # compute gradient for weight_mask
        self.sparse_layer.weight_mask.requires_grad = True

    """
    after configuring the mask, it's important to update and allow gradient to pass through the weight for training
    """
    def revert_weight_grad_and_update_mask(self, mask=None):
        # Turn back on the gradient for weight
        self.sparse_layer.weight.requires_grad = True
        # update mask
        if mask!=None:
            del self.sparse_layer.weight_mask
            if self.mask_cat=='scatter':
                self.keep_mask_idx = torch.where(mask.flatten()==1)[0].to(self.sparse_layer.weight.device)
                self.sparse_layer.weight_mask = torch.zeros_like(self.sparse_layer.weight)
                self.sparse_layer.weight_mask.flatten().scatter_add_(0, self.keep_mask_idx, torch.ones(self.keep_mask_idx.shape).to(self.sparse_layer.weight.device))
            else:
                self.sparse_layer.weight_mask = mask.to(self.sparse_layer.weight.device)
        else:
            print('Mask is not provided, initializing to default mask value=1')
            del self.sparse_layer.weight_mask
            self.sparse_layer.weight_mask = torch.ones(self.sparse_layer.weight_mask.shape).to(self.sparse_layer.weight.device)

        



# @register_modifier("scatter_sparse_mask_adapter", config_cls=SparseMaskConfig)
# class ScatterSparseMaskAdapter(ModifyMixin):
#     def __init__(
#         self,
#         config: SparseMaskConfig,
#         layer: nn.Module,
#         **kwargs,
#     ):
#         super().__init__()

#         self.layer=layer
#         input_dim, output_dim = self.layer.weight.T.shape
#         self.param_shape = self.layer.weight.shape
#         self.param_num = self.layer.weight.numel()
        
#         # weight initialization
#         self.sparse_layer = nn.Linear(input_dim, output_dim).to(device=layer.weight.device)
#         self.sparse_layer.weight = nn.Parameter(torch.zeros(self.sparse_layer.weight.shape))
#         self.sparse_layer.bias = nn.Parameter(torch.zeros(self.sparse_layer.bias.shape))

#         self.BLOCK_SIZE = config.BLOCK_SIZE
#         self.BlockwiseConvolution = MatrixBlockIndexer(M=input_dim, N=output_dim, BLOCK_SIZE=self.BLOCK_SIZE)
   
#         # mask initialization
#         self.sparse_layer.weight_mask = torch.ones(self.sparse_layer.weight.shape).to(device=layer.weight.device)
#         self.mask_cat= config.mask_cat
#         self.keep_ratio = config.keep_ratio
#         self.keed_mask_idx = None                 # will be initialized during training        

#         self.sparse_cat = config.sparse_cat 
#         assert self.sparse_cat in ['block_sparse','regular_sparse'], "Choose `sparse_cat` from ['block_sparse','regular_sparse'] " 

#         self.training_mode = config.training_mode
#         # noise initialization
#         self.activate_noise = config.activate_noise    # if True, uses a noisy parameter training
#         assert type(self.activate_noise) == bool

#         # update forward step
#         if self.training_mode == False:
#             pass
#             # forward function
#             # self.patch_forward()                
#         else:
#             # forward function        
#             if self.activate_noise:
#                 # 'nextK_idx' will be initialized during training, used in adding noise
#                 # (a) if noise_cat=='targeted_noise': stores the idx of most % important weights after the trainable params
#                 # (b) if noise_cat=='random_noise': a larger-matrix keeps idx of everything except the trainable params
#                 self.nextK_idx = None
#                 self.noise_cat = config.noise_cat             
#                 assert self.noise_cat in ['targeted_noise','random_noise']

#                 self.noise_space_ratio = config.noise_space_ratio # % of parameter space that we sample noise idx from
#                 self.noise_add_ratio = config.noise_add_ratio  # % of parameter where noise will be added

#                 assert (self.noise_space_ratio > self.keep_ratio and self.noise_space_ratio > self.noise_add_ratio), "since noise indx is sampled from `noise_space_ratio`, it's value must be greater than `keep_ratio` and `noise_space_ratio` "
                
#                 # TODO: I don't think we need two matrix here
#                 #self.zero_matrix = torch.zeros(self.param_shape, device=self.layer.weight.device)
#                 self.sparse_layer.noise = torch.zeros(self.param_shape, device=self.layer.weight.device)
#                 self.turn_off_require_grad()


#                 # self.patch_noisy_forward()
#             else:
#                 # self.patch_forward()
#                 pass
#         # used for noise calculcation
#         self.sampled_indices = None 
#         self.noise = None

#     def confirm_sync_device(self):
#         with torch.no_grad():
#             # sync the device of the `weight_mask` with module layers
#             if self.sparse_layer.weight_mask.requires_grad:
#                 # when gradient graph is true
#                 if self.sparse_layer.weight.device!=self.layer.weight.device:
#                     self.sparse_layer.weight_mask = self.sparse_layer.weight_mask.to(self.layer.weight.device)
#             else:
#                 # otherwise
#                 # required `.clone()` before device transfer, otherwise getting following error
#                 # """RuntimeError: Inference tensors cannot be saved for backward. 
#                 #    To work around you can make a clone to get a normal tensor and use it in autograd."""
#                 #if self.sparse_layer.weight_mask.device!=self.layer.weight.device:
#                 # self.sparse_layer.weight_mask = self.sparse_layer.weight_mask.clone().to(self.layer.weight.device)
#                 weight_mask = self.sparse_layer.weight_mask.to(self.layer.weight.device)
#                 self.sparse_layer.weight_mask = weight_mask
        

#     def forward(self, input):
#         # `weight_mask` requires to sync with `weight` device, as default only looks module 
        
#         self.confirm_sync_device()  # TODO, need to remove this, it should only require once before training

#         weight = self.layer.weight.flatten().scatter_add(0, self.sparse_layer.weight_mask, self.sparse_layer.weight)
#         weight = weight.to(self.layer.weight.device)
#         weight = weight.reshape_as(self.layer.weight)

#         return torch.nn.functional.linear(input, weight, self.layer.bias + self.sparse_layer.bias)



def make_sparse_model(module, dm, keep_ratio=0.05):
    """
    useful for quick check and prototype, not used in the training
    """
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule

    # (1) preprocess the sparse-layers
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.preprocess_for_mask_update()

    # (2) collect grads
    data_iter = iter(dm.train_dataloader())
    batch = next(data_iter)
    from mttl.models.utils import transfer_batch_to_device
    module = module.to('cuda')
    batch = transfer_batch_to_device(batch, module.device)
    loss = module.forward(batch)
    loss.backward()

    # (3) compute mask
    # (a) layer-wise
    mask_indx = []
    save_mask_indx = True
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            #m.sparse_layer.weight_mask.grad
            num_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask)*keep_ratio)
            threshold, _ = torch.topk(m.sparse_layer.weight_mask.grad.flatten(), num_params_to_keep, sorted=True)
            accepted_score = threshold[-1]
            keep_masks = (m.sparse_layer.weight_mask.grad>=accepted_score).float()
            # (4) revert back to original state
            # (a) reverse the require-grad: Turn on for `weight` and turn-off for `weight_mask`
            # (b) convert `module` back to `cpu`
            m.revert_weight_grad_and_update_mask(keep_masks)
            
            #TODO: generate noise:
            m.generate_noise()

            #if save_mask_indx: mask_indx.append(torch.nonzero(keep_masks).data.cpu().numpy()) # nonzero finds the ind
    # (b) based on whole-net TODO
    module = module.to('cpu')
    # if save_mask_indx: 
    #     import h5py
    #     import os
    #     # Ensure the directory exists or create it if not
    #     os.makedirs('saved_masks', exist_ok=True)
    #     f_name = f'saved_masks/{dm.config.finetune_task_name}'
    #     np.savez_compressed(f'{f_name}.npz', arr=mask_indx)
    #     # with h5py.File(f'{f_name}.h5', 'w') as f:
    #     #     f.create_dataset('data', data=mask_indx, compression='gzip')
    print('done')




def compress_sparse_2D_weight(module):
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.convert_sparse_weight_to_1D()




def convert_2D_idx_mask(weight_idx, mat_dim):
    m = np.zeros(mat_dim)
    m[tuple(zip(*weight_idx))] = 1
    return torch.FloatTensor(m)
def convert_2D_idx_1D(weight_idx, cols):
    row_idx, col_idx = weight_idx[:,0], weight_idx[:,1]
    index_1d = row_idx * cols + col_idx
    return torch.tensor(index_1d)


def load_scatter_pretrain_model(module, args):

    # load mask
    from huggingface_hub import hf_hub_download
    import numpy as np
    destination_type, f_name = args.library_id.split('://')
    repo_id=('/').join(f_name.split('/')[:2])
    filename = f'{args.task_names[0]}_mask.npz'
    f_path=hf_hub_download(repo_id=repo_id, filename=filename)
    Mask = np.load(f_path, allow_pickle=True)['arr'].item()

    # load params
    checkpoint = 'test_phi2_sparse_finetune/wiki_hop_original_explain_relation_1/epoch_0.ckpt'
    sparse_weights = torch.load(checkpoint)["state_dict"]
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule
    from mttl.models.modifiers.sparse_mask import ScatterSparseMaskAdapter as ScatterSparseMaskModule
    for m_name, m in dict(module.named_modules()).items():
        if isinstance(m, SparseMaskModule) or isinstance(m, ScatterSparseMaskModule):
            del m.sparse_layer.weight
            del m.sparse_layer.bias
            device = m.layer.weight.device
            new_weight = torch.nn.Parameter(sparse_weights[f'{m_name}.sparse_layer.weight'].data.clone().to(device=device, dtype=torch.bfloat16), requires_grad=True)
            m.sparse_layer.weight = new_weight
            new_bias = torch.nn.Parameter(sparse_weights[f'{m_name}.sparse_layer.bias'].data.clone().to(device=device, dtype=torch.bfloat16), requires_grad=True)
            m.sparse_layer.bias = new_bias
            keep_mask = convert_2D_idx_1D(weight_idx=Mask[f'{m_name}.sparse_layer'],
                                        cols=m.sparse_layer.weight_mask.shape[1]).to(device)
            

            m.sparse_layer.weight_mask=keep_mask.data.clone()
    del sparse_weights


import json
def compute_noise_mean_std(exp='phi-3_regular_sparse_kr_0.1'):
    """
    based on sparse-trained model, we compute the mean-std weights over different layers
    """

    with open(f'Weight_Stats/{exp}/weight_stats.json') as f:
        d = json.load(f)
        tasks=list(d.keys())
        layers = list(d[tasks[0]].keys())
        

        noise_mean = {}
        noise_std = {}
        for l in layers:
            noise = []
            for t in tasks:
                noise.append(d[t][l][0])
            noise_mean[l] = np.mean(noise)
            noise_std[l] = np.std(noise)
    noise_mean = np.array(list(noise_mean.values()))
    noise_std = np.array(list(noise_std.values()))
    
    layers = ['.'.join(l.split('.')[:-1]) for l in layers]
    m = dict(zip(layers,noise_mean))
    s = dict(zip(layers,noise_std))
    return m, s



def apply_fixed_noise_to_sprase_module(module, exp='phi-3_regular_sparse_kr_0.1'): 
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule
    noise_mean, noise_std = compute_noise_mean_std(exp)
    for m_name, m in dict(module.named_modules()).items():
        if isinstance(m, SparseMaskModule):
            m_name = '.'.join((m_name.split('.'))[1:])
            m.noise_mean = noise_mean[m_name]
            m.noise_std = noise_std[m_name]
    print('added fixed noise')