import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="experiment/gsm.jsonl")
args = parser.parse_args()


def extract_code(text):
    match = re.search(r"```(.*?)```", text, re.DOTALL)

    code_block = re.search(r"def solution\(\):.*?\n\n", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    elif code_block:
        code_block = code_block.group(0)
        return code_block
    return None


correct = 0
all_count = 0
invalid_code = 0
error_execute = 0
with open(args.file, "r") as f:
    for line in f:
        all_count += 1
        data = json.loads(line)
        answer = data["answer"]
        code = extract_code(data["output_pred"])
        if code is None:
            invalid_code += 1
            continue
        predict_answer = None
        try:
            exec(code)
            exec("predict_answer = solution()")
            # exec("print(predict_answer, answer)")
            # compute the accuracy
        except Exception as e:
            error_execute += 1
            print(e)
        if predict_answer == answer:
            correct += 1
print(correct, all_count)
print("Accuracy:", correct / all_count)
print("invalid_code", invalid_code)
print("error_execute", error_execute)
