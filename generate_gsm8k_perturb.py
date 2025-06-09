import os
from openai import AzureOpenAI
from datasets import load_dataset
import json
from tqdm import tqdm

endpoint = "https://proj-smart-summer-2025-resource.cognitiveservices.azure.com/"
model_name = "gpt-4.1"
deployment = "gpt-4.1"
key = os.environ["OPENAI_API_KEY"]
subscription_key = key
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


def get_response(question):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


# load the gsm8k dataset

dataset = load_dataset("openai/gsm8k", "main")
dataset = dataset["test"]

insturction = "You are a high school math teacher. Now you need to judge if the student answer the question based on memory or based on reasoning. \
So please perturb the only one number of the question and then give the answers. \
For example, here is the original question: \
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\
#### 18\
Now you need to perturb the only one number of the question and then give the answers. \
For example, here is the perturbed question: \
<Question>Janet’s ducks lay 20 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? </Question> \
Janet sells 20 - 3 - 4 = <<20-3-4=>>13 duck eggs a day.\
She makes 13 * 2 = $<<13*2=26>>26 every day at the farmer’s market.\
<Answer>26</Answer> \
You should not change the format and the answer format. based on my example. Now you need to perturb the only one number of the question and then give the answers.{}"

# result = get_response(insturction.format(dataset[0]["question"] + dataset[0]["answer"]))
# breakpoint()

file = open("result.jsonl", "w")
for i in tqdm(range(len(dataset))):
    question = dataset[i]["question"]
    answer = dataset[i]["answer"]
    result_open_ai = get_response(insturction.format(question + answer))

    question_open_ai = result_open_ai.split("<Question>")[1].split("</Question>")[0]
    answer_open_ai = result_open_ai.split("<Question>")[1].split("</Question>")[1]
    json_print = {
        "question": question_open_ai,
        "answer": answer_open_ai,
        "original_question": question,
        "original_answer": answer,
    }

    file.write(json.dumps(json_print) + "\n")
    file.flush()
