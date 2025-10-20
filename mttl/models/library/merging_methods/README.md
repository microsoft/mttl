This repository provides implementations of various model merging techniques:

### Weight Averaging `UniformMerge`
Supports averaging weights from different finetuning strategies, including:
- Dense finetuning  
- LoRA  
- Sparse adapters  

### Uniform Sparse `UniformSparse`
* Performs more accurate merging for sparse adapters by considering parameter overlap, described in the paper: *[Exploring Sparse Adapters for Scalable Merging of Parameter Efficient Experts](https://openreview.net/forum?id=8wt2eKkVe6)*

### TIES `TiesMergeSimple`
* Implements the technique from the paper: *[TIES: Task-Interpolated Expert Selection](https://arxiv.org/pdf/2306.01708)*

### SLERP (Spherical Linear Interpolation) `slerp`
* Implementation adapted from: *[LLM-SLERP-Merge](https://github.com/Digitous/LLM-SLERP-Merge/blob/main/slerpmergelm.py)*

### Task Arithmetic `TaskArithmetic`
* Current implementation computes task-vectors and only allow averaging
* Based on the method described in: *[Task Arithmetic Paper](https://openreview.net/pdf?id=6t0Kwf8-jrj)*

### Model Breadcrumbs `TaskArithmetic`
* Implements the approach from: *[Model Breadcrumbs Paper](https://arxiv.org/pdf/2312.06795)*