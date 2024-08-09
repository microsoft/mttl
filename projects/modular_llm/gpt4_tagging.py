import asyncio
import json
import os
from collections import defaultdict

import fire
import jinja2
import numpy as np
import tenacity
from openai import AsyncAzureOpenAI
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm

client = AsyncAzureOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    azure_endpoint="https://gcrgpt4aoai7.openai.azure.com/",
    api_version="2024-05-01-preview",
)


gpt_model = None
g_infer_model = "gpt-4o-mini"


m_template = """
The following instructions are associated with a specific tag. Your task is to create a precise and descriptive title for this tag, encapsulating a common aspect found in most of these instructions.

{% for instruction in instructions %}
Instruction:
{{instruction}}

{% endfor %}

Determine a better title that encapsulates a common aspect found in most of these instructions, please provide it in this format:
Tag: Descriptive title for the tag

"""


m_contrastive_template = """
The following two groups of instructions A and B are each associated with a specific tag. Your task is to create a precise and descriptive title for the tag of group B, encapsulating a common aspect found in most of these instructions.
The title should describe the instructions in group B in a way that contrasts with common aspects of the instructions in group A.

{% for instruction in instructions_a %}
Instruction (Group A):
{{instruction}}

{% endfor %}

{% for instruction in instructions_b %}
Instruction (Group B):
{{instruction}}

{% endfor %}

Determine a better title for the instructions in group B that encapsulates a common aspect found in most of these instructions while not encapsulating common aspects of the instructions in group A, please provide it in this format:
Tag: Descriptive title for the tag

"""


e_template = """
Select a tag from the list that most accurately represents the given instruction. If a specific tag accurately describes the instruction, prioritize it over a more generic one.

{{instruction}}

Tags:
{% for tag in tags %}
{{loop.index}}. {{tag}}
{% endfor %}

Please indicate the most appropriate tag by providing ONLY the corresponding number (1-{{tags|length}}). Use the following format:
Tag number: Number

Tag number:"""


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=10, max=60),
    stop=tenacity.stop_after_attempt(100),
)
async def get_completions(prompt, num_completions=1, max_tokens=128):
    response = await client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stop="\n",
        temperature=0.0,
        max_tokens=max_tokens,
        n=num_completions,
    )
    return [choice.message.content for choice in response.choices]


async def get_tag(instruction):
    return await get_new_tag([instruction])


async def get_new_tag(instructions):
    response = await get_completions(
        jinja2.Template(m_template).render(instructions=instructions)
    )
    response = response[0]
    if response.startswith("Tag:"):
        return response[len("Tag:") :].strip().rstrip(".").strip()
    return None


async def get_new_tag_contrastive(instructions_a, instructions_b):
    response = await get_completions(
        jinja2.Template(m_contrastive_template).render(
            instructions_a=instructions_a, instructions_b=instructions_b
        )
    )
    response = response[0]
    if response.startswith("Tag:"):
        return response[len("Tag:") :].strip().rstrip(".").strip()
    return None


async def assign_tag(instruction, tags):
    if len(instruction.split()) >= 50_000:
        print("too long instruction, skipping.")
        return None

    response = await get_completions(
        jinja2.Template(e_template).render(instruction=instruction, tags=tags),
        max_tokens=5,
    )
    response = response[0]
    try:
        # parse response right away
        return int(response.strip()) - 1
    except:
        # if the model has generated "tag number"...
        if response.startswith("Tag number:"):
            try:
                return int(response[len("Tag number:") :].strip()) - 1
            except:
                return None
    return None


def get_instructions(examples, return_metadata=False):
    """Returns user instructions as a stream of messages.

    Optionally returns associated metadata indicating the turn number and whether the message is the last in the example.
    """

    def load_messages(line):
        # we load the line as a json object
        example = json.loads(line)
        messages = example["messages"]

        yield from [
            (turn, m["content"])
            for turn, m in enumerate(messages)
            if m["role"] == "user"
        ]

    for example_id in range(len(examples)):
        messages = list(load_messages(examples[example_id]))
        for i, (turn, message) in enumerate(messages):
            if return_metadata:
                # message, example_index, turn, is last from this example id
                is_last = i == len(messages) - 1
                yield message, example_id, turn, is_last
            else:
                yield message


def get_batch(dataloader, batch_size):
    batch = []
    for _ in range(batch_size):
        try:
            batch.append(next(dataloader))
        except StopIteration:
            return []
    return batch


async def train_jsonl_file(file_path, output_path, num_tags, mode="normal"):
    """E-M training for tagging instructions."""
    import random

    random.seed(42)

    tags = None
    batch_size = num_tags * 15
    init_examples_per_tag = 5

    with open(file_path, "r") as ifile:
        train_examples = ifile.readlines()
        random.shuffle(train_examples)

    train_loader = iter(get_instructions(train_examples))

    # initialize the tags here, 10 examples per tag
    if tags is None:
        init_examples = [
            get_batch(train_loader, init_examples_per_tag) for _ in range(num_tags)
        ]
        tags = await tqdm.gather(*[get_tag(c) for c in init_examples])

        for i, tag in enumerate(tags):
            print(f"{i}.", tag)

    iteration = 0
    ofile = open(output_path, "w")
    while iteration < 30:
        batch = get_batch(train_loader, batch_size)
        if not batch:
            break

        # now get tags for the batch
        tagged_examples = await tqdm.gather(
            *[assign_tag(example, tags) for example in batch]
        )

        # now group examples by tags, not that some examples may not have tags
        # at this point, so some tags might not be useful!
        notag_examples = []
        grouped_examples = defaultdict(list)
        for example, tag in zip(batch, tagged_examples):
            if tag is None:
                notag_examples.append(example)
                continue
            grouped_examples[int(tag)].append(example)

        # print "entropy" of the distribution, normalize the length of each group and compute the normalized entropy
        lengths = [len(v) for v in grouped_examples.values()]
        norm_lengths = np.array(lengths) / np.sum(lengths)
        entropy = -np.sum(norm_lengths * np.log(norm_lengths + 1e-6)) / np.log(
            len(lengths)
        )
        print("Entropy of tags assignments: ", entropy)

        # get "widow" tags, create a new tag for each group, here we sample random examples
        # until each group has 10 examples
        for i in range(num_tags):
            if i not in grouped_examples or len(grouped_examples[i]) == 1:
                grouped_examples[i].extend(
                    get_batch(train_loader, 10 - len(grouped_examples[i]))
                )

        # now get the m-step for each group
        groups, keys = [], []
        for key, group in grouped_examples.items():
            # cap maximum group size
            if len(group) >= 10:
                group = group[:10]

            groups.append(group)
            keys.append(key)

        print("M-step for # tags =", len(groups))

        # in normal mode we just sample conditioned on the current group
        if mode == "normal":
            new_tags = await tqdm.gather(*[get_new_tag(group) for group in groups])
        # in contrastive mode, we sample a negative group the tag should *not* describe
        elif mode == "contrastive":
            print("Contrastive mode")
            args = []
            for i in range(len(groups)):
                j = np.random.choice([k for k in range(len(groups)) if k != i])
                args.append((groups[j], groups[i]))

            new_tags = await tqdm.gather(
                *[
                    get_new_tag_contrastive(group_a, group_b)
                    for group_a, group_b in args
                ]
            )
        else:
            raise ValueError("Invalid mode:", mode)

        new_tags_dict = dict(zip(keys, new_tags))

        new_tags = []
        for i in range(num_tags):
            if i in new_tags_dict:
                line = "{:<100} {:>15}".format(
                    new_tags_dict[i], f"{len(grouped_examples[i])}/{batch_size} (*)"
                )
                print(f"{i}.", line)
                new_tags.append(new_tags_dict[i].strip())
            else:
                print(f"{i}.", "{:<100}".format(tags[i].strip()))
                new_tags.append(tags[i].strip())

        tags = new_tags
        iteration += 1

        ofile.write(json.dumps({"iteration": iteration, "tags": tags}) + "\n")
        ofile.flush()
    ofile.close()


async def infer_jsonl_file(file_path, tags_file, output_path, num_inferences=-1):
    tags = None
    end = False
    batch_size = 100

    # load the tags
    with open(tags_file, "r") as ifile:
        tags = json.loads(ifile.readlines()[-1])["tags"]

        for i, t in enumerate(tags):
            print(f"{i}.", t.strip())

    # load all the examples to be tagged
    with open(file_path, "r") as ifile:
        train_examples = ifile.readlines()
        train_loader = get_instructions(train_examples, return_metadata=True)

    try:
        with open(output_path, "r") as ofile:
            resume_from_line = ofile.readlines()
            resume_from = len(resume_from_line)
            print("Resuming from example id:", resume_from)
    except:
        resume_from = 0

    done = 0
    ofile = open(output_path, "w")
    progress_bar = ttqdm(total=len(train_examples) + 100_000)
    while not end and (num_inferences == -1 or done < num_inferences):
        batch = []
        metadata = []
        try:
            for message, ex_id, turn, is_last in train_loader:
                if ex_id >= resume_from:
                    batch.append(message)
                    metadata.append((ex_id, turn, is_last))
                else:
                    if is_last:
                        progress_bar.update(1)

                if len(batch) == batch_size:
                    break
        except StopIteration:
            end = True

        # now get tags for the batch
        tagged_examples = await tqdm.gather(
            *[assign_tag(example, tags) for example in batch]
        )

        n_random = 0
        for i, tag in enumerate(tagged_examples):
            if tag is None:
                tag = np.random.randint(len(tags))
                n_random += 1

            # now we have to inject the tags back in the original dataset
            example_id, turn, is_last = metadata[i]
            example = json.loads(train_examples[example_id])
            messages = example["messages"]
            # tag this particular message with its corresponding cluster
            messages[turn]["tag"] = tags[int(tag)]
            train_examples[example_id] = json.dumps(example)
            # if it is the last message for this example, write it to the file
            if is_last:
                ofile.write(train_examples[metadata[i][0]] + "\n")
                done += 1

        progress_bar.update(len(batch))
        print("Random tags:", n_random)
    ofile.close()


async def train_(json_file_path, output_path, num_tags=100, mode="normal"):
    global gpt_model
    global g_infer_model

    await train_jsonl_file(
        json_file_path,
        output_path,
        num_tags=num_tags,
        mode=mode,
    )

    # switch inference mode
    gpt_model = g_infer_model
    await infer_(
        json_file_path,
        output_path,
        num_inferences=10_000,
    )


async def infer_(json_file_path, tags_file, num_inferences=-1):
    await infer_jsonl_file(
        json_file_path,
        tags_file,
        tags_file.replace(".jsonl", "")
        + f"_inferred_{gpt_model}_n{num_inferences}.jsonl",
        num_inferences=num_inferences,
    )


class GPT4EMTagging:
    def infer(self, tags_path, file_path, model="gpt-4o-mini", num_inferences=-1):
        global gpt_model

        gpt_model = model

        print("Working on...", file_path)
        asyncio.run(infer_(file_path, tags_path, num_inferences=num_inferences))

    def train(
        self,
        file_path,
        output_path,
        num_tags=100,
        model="gpt-4o-gs",
        infer_model="gpt-4o-mini",
        mode="normal",
    ):
        global gpt_model
        global g_infer_model

        gpt_model = model
        g_infer_model = infer_model

        print("Working on...", file_path)
        asyncio.run(train_(file_path, output_path, num_tags=num_tags, mode=mode))


if __name__ == "__main__":
    fire.Fire(GPT4EMTagging)
