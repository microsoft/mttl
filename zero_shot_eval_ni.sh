###########   Purpose: This script is used to evaluate the zero-shot performance of the model on the NI dataset.

# generated the prediction file
CUDA_VISIBLE_DEVICES=1 nohup python gen_ni_predictions_bak.py --out_prefix=cluster_test_bert --batch_size=1 --example_to_ids_path=inst_follow/data/cluster_infos/atlas_by_instr_bert-base-uncased_ldalayer2.pkl > cluster_test_bert.log 2>&1 &

# evaluate the prediction file
# python evaluate.py --prediction_file=ni_pred_test_yahma_llama-7b-hfni-nshot_canonical.jsonl --reference_file=test_references.jsonl