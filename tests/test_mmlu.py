from mttl.dataloader.mmlu_dataset import format_example_with_augmentation


def test_mmlu_augmentation():
    icl_prompts = ["PROMT_ICL"]
    icl_options = [["0", "1", "2", "3"]]
    icl_labels = ["B"]

    prompt = "PROMT"
    options = ["4", "8", "12", "6"]
    label = "A"

    gen = format_example_with_augmentation(
        prompt,
        options,
        label,
        icl_prompts,
        icl_options,
        icl_labels,
        prompt_def="blabla",
        augment_with_prompts=True,
        augment_with_options=True,
    )
    outs = []
    for _prompt_end, _label, _prompt_def, _prompt_pos in gen:
        outs.append((_prompt_end, _label, _prompt_def, _prompt_pos))
    assert len(outs) == 7
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[0]
    # prompt augmented
    assert _prompt_end == "Question:\nPROMT\nChoices:\nA. 4\nB. 8\nC. 12\nD. 6\nAnswer:"
    assert _label == "A"
    assert _prompt_def == ""
    assert (
        _prompt_pos
        == "Question:\nPROMT_ICL\nChoices:\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"
    )

    # no prompt augmented, options augmented 1
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[1]
    assert _prompt_end == "PROMT\nA. 8\nB. 4\nC. 12\nD. 6\nAnswer:"
    assert _label == "B"
    assert _prompt_def == "blabla"
    assert _prompt_pos == "PROMT_ICL\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"

    # prompt augmented, options augmented 1
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[2]
    assert _prompt_end == "Question:\nPROMT\nChoices:\nA. 8\nB. 4\nC. 12\nD. 6\nAnswer:"
    assert _label == "B"
    assert _prompt_def == ""
    assert (
        _prompt_pos
        == "Question:\nPROMT_ICL\nChoices:\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"
    )

    # no prompt augmented, options augmented 2
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[3]
    assert _prompt_end == "PROMT\nA. 12\nB. 8\nC. 4\nD. 6\nAnswer:"
    assert _label == "C"
    assert _prompt_def == "blabla"
    assert _prompt_pos == "PROMT_ICL\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"

    # prompt augmented, options augmented 2
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[4]
    assert _prompt_end == "Question:\nPROMT\nChoices:\nA. 12\nB. 8\nC. 4\nD. 6\nAnswer:"
    assert _label == "C"
    assert _prompt_def == ""
    assert (
        _prompt_pos
        == "Question:\nPROMT_ICL\nChoices:\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"
    )

    # no prompt augmented, options augmented 3
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[5]
    assert _prompt_end == "PROMT\nA. 6\nB. 8\nC. 12\nD. 4\nAnswer:"
    assert _label == "D"
    assert _prompt_def == "blabla"
    assert _prompt_pos == "PROMT_ICL\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"

    # prompt augmented, options augmented 3
    _prompt_end, _label, _prompt_def, _prompt_pos = outs[6]
    assert _prompt_end == "Question:\nPROMT\nChoices:\nA. 6\nB. 8\nC. 12\nD. 4\nAnswer:"
    assert _label == "D"
    assert _prompt_def == ""
    assert (
        _prompt_pos
        == "Question:\nPROMT_ICL\nChoices:\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"
    )

    gen = format_example_with_augmentation(
        prompt,
        options,
        label,
        icl_prompts,
        icl_options,
        icl_labels,
        prompt_def="blabla",
        augment_with_prompts=False,
        augment_with_options=False,
    )
    outs = []
    for _prompt_end, _label, _prompt_def, _prompt_pos in gen:
        outs.append((_prompt_end, _label, _prompt_def, _prompt_pos))
    assert len(outs) == 0

    gen = format_example_with_augmentation(
        prompt,
        options,
        label,
        icl_prompts,
        icl_options,
        icl_labels,
        prompt_def="blabla",
        augment_with_prompts=False,
        augment_with_options=True,
    )
    outs = []
    for _prompt_end, _label, _prompt_def, _prompt_pos in gen:
        outs.append((_prompt_end, _label, _prompt_def, _prompt_pos))
    assert len(outs) == 3

    gen = format_example_with_augmentation(
        prompt,
        options,
        label,
        icl_prompts,
        icl_options,
        icl_labels,
        prompt_def="blabla",
        augment_with_prompts=True,
        augment_with_options=False,
    )
    outs = []
    for _prompt_end, _label, _prompt_def, _prompt_pos in gen:
        outs.append((_prompt_end, _label, _prompt_def, _prompt_pos))
    assert len(outs) == 1
