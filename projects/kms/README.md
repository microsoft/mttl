

## Generate data with gpt-4o-mini

First, set environment variables OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION. Then if you want to generate summaries:

```python generate_for_wikipedia.py --model gpt-4o-mini --output_path wiki_summ_gpt-4o-mini --use_prompts summary```

Else, if summaries and qa are needed:

```python generate_for_wikipedia.py --model gpt-4o-mini --output_path wiki_summ-qa_gpt-4o-mini --use_prompts summary,qa```

If we want to use a specific model:

```python generate_for_wikipedia.py --model microsoft/Phi-3-medium-4k-instruct --output_path wiki_summ_phi-3_medium --use_prompts summary```

This will create a HF dataset, that can be loaded with `datasets.load_from_disk`. The dataset contains the following columns:

```
['input', 'outputs', 'subject', 'type']
```

`input` is the chunk of text, `outputs` is a list of generations, `subject` is the corresponding `mmlu` subject, `type` is wheter it's `summary` or `qa`.

To read the dataset, we can just use `KMDataloader`.

To train the KMs, just use `python train_km.py -c phi-3.json -k dataset=... -k finetune_task_name=MMLU_SUBJECT expert_name=MMLU_SUBJECT`

### Train KMs on NarrativeQA

Assuming generated data under `./nqa_summary_data`

1. Create a "compact" version of the dataset by `python create_nqa_dataset.py --hf_id your_hf_id/narrativeqa`

2. Use `python train_km.py -c phi-3.json -k dataset=local://nqa_summary_data -k finetune_task_name=NQA_DOCUMENT_ID task_name_field=document_id nqa_dataset=your_hf_id/narrativeqa`

`nqa_dataset` might be left `None` if one doesn't want zero-shot eval during training.

### Preliminary Reproduction results 
Note : KEs we trained on a subset of 100 training documents (commit `b1a47e98f610b819373377285942ad528bb3473d`) 

1. Phi-3 + KE = 19.92 
2. Phi-3 + IC summary + KE = 75.94
4. Phi-3 + RAG + KE = 33.16
3. Phi-3 + IC summary =  31.94