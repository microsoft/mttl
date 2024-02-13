for cluster in "c0o10_2e" "c1o10_2e";
do
python train_experts_main.py \
      -c configs/wiki-mmlu/gpt2neo_1B_experts.json+configs/wiki-mmlu/gpt2neo_1B_dense.json \
      -k output_dir=~/mttl_out/ \
      dataset=sordonia/flan-10k-flat \
      predict_batch_size=32 \
      finetune_task_name=${cluster} \
      num_train_epochs=3 \
      learning_rate=3e-4 \
      pipeline_eval_tasks=all \
      eval_every_n_epoch=None \
      create_transfer_matrix=True \
      eval_every=400 \
      tasksets_path=task_sets/gpt_neo1B/gptneo_1B_clusters.json
done