import json
import os
import uuid
from threading import Thread

import torch
from flask import Flask, jsonify, render_template, request, session
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.models.base_model import AutoExpertModel
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.modifiers.lora import LoRAConfig

app = Flask(__name__)
app.secret_key = "your_secret_key"

os.makedirs("./logs", exist_ok=True)

# load model and all the experts
model = MultiExpertModel(
    MultiExpertModelConfig(base_model="microsoft/Phi-3-mini-4k-instruct"),
    device_map="cuda:0",
)
model.add_experts_from_library("local://../library_km_wiki_phi-3_medium/")
# hack: add an expert corresponding to the base model
one_expert = next(iter(model.experts_infos.values()))
model.add_empty_expert("None", expert_config=one_expert.expert_config, is_default=False)

tokenizer = get_tokenizer_with_args(
    model.config.base_model,
    model_family="gpt",
    padding_side="left",
    truncation_side="left",
    for_generation=True,
)

conversation = {}
active_modules = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_knowledge_module", methods=["POST"])
def load_knowledge_module():
    data = request.json
    km_name = data.get("module_name")
    window_id = data.get("window_id", None)

    if window_id not in active_modules:
        active_modules[window_id] = None

    try:
        active_modules[window_id] = km_name
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False})


@app.route("/clear", methods=["POST"])
def clear():
    from flask import Response, request, stream_with_context

    data = request.json
    window_id = data.get("window_id")

    if window_id not in conversation:
        conversation[window_id] = []

    conversation[window_id].clear()
    return jsonify({"success": True})


@app.route("/send", methods=["POST"])
def send():
    from flask import Response, request, stream_with_context

    data = request.json
    message = data.get("message")
    window_id = data.get("window_id")

    if window_id not in conversation:
        conversation[window_id] = []

    if window_id not in active_modules:
        active_modules[window_id] = "None"

    conversation[window_id].append({"role": "user", "content": message})

    def generate(message):
        task_names = [active_modules[window_id]]
        generation_streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        generation_kwargs = dict(
            input_ids=tokenizer.apply_chat_template(
                conversation[window_id], return_tensors="pt", add_generation_prompt=True
            ).to(model.device),
            task_names=task_names,
            streamer=generation_streamer,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        text = ""
        for i, outputs in enumerate(generation_streamer):
            if not outputs or i == 0:
                continue
            text += outputs
            yield text

        conversation[window_id].append({"role": "assistant", "content": text})

        with open(f"./logs/{window_id}.jsonl", "w") as f:
            f.write(json.dumps(conversation[window_id]) + "\n")

    response = Response(generate(message), mimetype="text/event-stream")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
