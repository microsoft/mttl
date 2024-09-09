import os
import uuid
from threading import Thread

import torch
from flask import Flask, jsonify, render_template, request, session
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.models.base_model import AutoExpertModel
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

app = Flask(__name__)
app.secret_key = "your_secret_key"

os.makedirs("./logs", exist_ok=True)
library_path = "local://../library_km_wiki_phi-3_medium/"
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = MultiExpertModel(
    MultiExpertModelConfig(base_model=model_name), device_map="cuda:0"
)
model.add_experts_from_library(library_path)
model.disable_adapters()

tokenizer = get_tokenizer_with_args(
    model.config.base_model,
    model_family="gpt",
    padding_side="left",
    truncation_side="left",
    for_generation=True,
)
conversation = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_model", methods=["POST"])
def load_model():

    global model, tokenizer, model_name

    data = request.json
    model_name = data.get("model_name", "gpt2")

    try:
        model = AutoExpertModel.from_pretrained(model_name).cuda()
        tokenizer = get_tokenizer_with_args(
            model.base_model_name_or_path,
            model_family="gpt",
            padding_side="left",
            truncation_side="left",
            for_generation=True,
        )
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return jsonify({"success": False})


@app.route("/load_knowledge_module", methods=["POST"])
def load_knowledge_module():
    from mttl.datamodule.utils import get_tokenizer_with_args
    from mttl.models.base_model import AutoExpertModel

    global model, tokenizer, model_name

    data = request.json
    km_name = data.get("module_name")

    try:
        if km_name == "None":
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_default_expert(km_name)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False})


@app.route("/clear", methods=["POST"])
def clear():
    user_id = session.get("user_id", None)

    if not user_id:
        user_id = session["user_id"] = str(uuid.uuid4())
        conversation[user_id] = []

    if user_id not in conversation:
        conversation[user_id] = []

    conversation[user_id].clear()
    return jsonify({"success": True})


@app.route("/send", methods=["POST"])
def send():
    from flask import Response, request, stream_with_context

    data = request.json
    message = data.get("message")
    user_id = session.get("user_id", None)

    if not user_id:
        user_id = session["user_id"] = str(uuid.uuid4())
        conversation[user_id] = []

    if user_id not in conversation:
        conversation[user_id] = []

    conversation[user_id].append({"role": "user", "content": message})

    def generate(message):
        generation_streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        generation_kwargs = dict(
            input_ids=tokenizer.apply_chat_template(
                conversation[user_id], return_tensors="pt", add_generation_prompt=True
            ).to(model.device),
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
            yield outputs

        conversation[user_id].append({"role": "assistant", "content": text})

        import json
        import os

        with open(f"./logs/{user_id}.log", "a+") as f:
            f.write(json.dumps(conversation[user_id]) + "\n")

    response = Response(generate(message), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
