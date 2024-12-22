# open the json file and read the data
import json
import re


def extract_code(text):
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


correct = 0
all_count = 0
with open("experiment/gsm.jsonl", "r") as f:
    for line in f:
        all_count += 1
        data = json.loads(line)
        answer = data["answer"]
        code = data["output_pred"]
        if code is None:
            continue
        predict_answer = None
        try:
            exec(code)
            exec("predict_answer = solution()")
            exec("print(predict_answer, answer)")
            # compute the accuracy
        except Exception as e:
            print(e)
        if predict_answer == answer:
            correct += 1
print("Accuracy:", correct / all_count)
