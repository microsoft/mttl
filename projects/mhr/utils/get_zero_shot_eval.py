import glob
import sys
import json
import os
import pandas as pd


for result in sys.argv[1:]:
    files = glob.glob(result + "/**/*.jsonl", recursive=True)
    results = []

    for file in files:
        config = os.path.dirname(file) + "/config.json"
        with open(config, "r") as f:
            config = json.load(f)

        with open(file, "r") as f:
            for line in f:
                acc = json.loads(line)["test/accuracy"]
                break

        task_name = config["finetune_task_name"]
        results.append({"task": task_name, "acc": acc})

    df = pd.DataFrame(results).sort_values("task")
    print(df)
    print(df["acc"].mean())
