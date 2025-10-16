# Zero-Shot
# torchrun --nproc_per_node=4 eval_qa.py -c eval/phi3-nqa-zs -k subsample_test=1000 library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/phi3-nqa-zs

# RAG Zero-Shot
# torchrun --nproc_per_node=4 eval_qa.py -c eval/phi3-nqa-rag256-k5 -k subsample_test=1000 library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/phi3-nqa-rag256-k5

# KE no library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa -k subsample_test=1000 force=True callback_during_training=False output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa

# KE rag no library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k subsample_test=1000 force=True output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-rag

# KE rag no library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k subsample_test=1000 force=True output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-rag

# KE rag with library
# torchrun --nproc_per_node=1 train_qa.py -c train/llama-ql-sdcd-fp-fixed-right+train/ke-ql-rag-learnw -k cpu_offload=True library_id=local:///data/alsordon/library-llama-ql-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-ql-sdcd-fp-fixed-right/ke-ql-rag-learnw-library

# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-learnw -k subsample_test=1000 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-learnw-library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k subsample_test=1000 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library

# More KMs + cpu offload
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-learnw -k subsample_test=1000 cpu_offload=True max_train_tasks=500 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-learnw-library-all
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k subsample_test=1000 cpu_offload=True max_train_tasks=500 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-all

# LLAMA stuff
# torchrun --nproc_per_node=4 eval_qa.py -c eval/llama-nqa-zs -k subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/llama-nqa-zs
# torchrun --nproc_per_node=4 eval_qa.py -c eval/llama-nqa-rag256-k5 -k subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/llama-nqa-rag256-k5
# torchrun --nproc_per_node=4 eval_qa.py -c eval/llama-nqa-rag256-k5 -k subsample_test=1000 output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/llama-nqa-rag256-k5-no-library
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k subsample_test=1000 force=True output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag

# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-learnw -k do_eval=True cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-learnw-library
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k do_eval=True cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library

# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k patience=1 cpu_offload=False subsample_test=1000 output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-k1 topk_context=1
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k patience=1 cpu_offload=False subsample_test=1000 output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-k2 topk_context=2
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k patience=1 cpu_offload=False subsample_test=1000 output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-k3 topk_context=3

torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag -k patience=5 cpu_offload=False subsample_test=1000 output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-k1 topk_context=1
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k micro_batch_size=1 patience=5 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k3 topk_context=3
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k micro_batch_size=1 patience=5 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k5 topk_context=5
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k micro_batch_size=1 patience=5 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k8 topk_context=8

# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k max_train_tasks=500 patience=1 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k1-t500 topk_context=1
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k max_train_tasks=500 patience=1 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k3-t500 topk_context=3
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k max_train_tasks=500 patience=1 cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k4-t500 topk_context=4

# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k1 topk_context=1
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k2 topk_context=2
# torchrun --nproc_per_node=4 train_qa.py -c train/llama-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k cpu_offload=True subsample_test=1000 library_id=local:///data/alsordon/library-llama-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/llama-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library-k3 topk_context=3

# KE rag with library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-learnw -k subsample_test=1000 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-learnw-library
# torchrun --nproc_per_node=4 train_qa.py -c train/phi3-nqa-sdcd-fp-fixed-right+train/ke-nqa-rag-learnw -k subsample_test=1000 force=True library_id=local:///data/alsordon/library-phi3-nqa-sdcd-fp-fixed-right output_dir=/data/alsordon/outputs/phi3-nqa-sdcd-fp-fixed-right/ke-nqa-rag-learnw-library
