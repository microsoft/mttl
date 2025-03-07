

LIB_ID=None
EXP_NAME=None
MERGING_METHOD=uniform
CUDA_VISIBLE_DEVICES=0 python3 projects/sparse_finetuning/eval_library.py \
            -k \
            seed=42 \
            output_dir=eval/${EXP_NAME} \
            library_id=$LIB_ID \
            merge_or_route=${MERGING_METHOD} \
            include_task_source=* \
            dataset=sordonia/flan-10k-flat \
            predict_batch_size=4 \
            pipeline_eval_tasks=in_distribution \
            eval_metric=rougeL