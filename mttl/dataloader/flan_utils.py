def encode_with_messages_format(example):
    message_text = ""
    intruction = "<|user|>\n" + example["user"].strip() + "\n"
    intruction += "<|assistant|>\n"
    output = example["assistant"].strip()
    message_text += intruction
    message_text += output
    message_text = message_text.strip()

    return {
        "input_text": intruction,
        "output_text": output,
        "full_text": message_text,
        "hash_text": message_text,
    }
