import argparse
from pathlib import Path


END_TOKEN = "<|endoftext|>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("sample_data/deepj-90m"))
    parser.add_argument("--pretrain-characters", type=int, default=8_600_000_000)
    parser.add_argument("--smoltalk-limit", type=int, default=460_000)
    parser.add_argument("--mmlu-limit", type=int, default=100_000)
    parser.add_argument("--gsm8k-limit", type=int, default=8_000)
    parser.add_argument("--arc-limit", type=int, default=8_000)
    parser.add_argument("--validation-limit", type=int, default=2_000)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-instructions", action="store_true")
    parser.add_argument("--seed", type=int, default=90)
    return parser.parse_args()


def datasets_module():
    try:
        import datasets
        return datasets
    except ImportError as error:
        raise SystemExit("Install training-data-requirements.txt first") from error


def dataset(name, subset, split):
    return datasets_module().load_dataset(name, subset, split=split, streaming=True)


def clean(value):
    return str(value).replace(END_TOKEN, "").replace("Response:\n", "Response: ").strip()


def record(instruction, response):
    return f"Instruction:\n{clean(instruction)}\nResponse:\n{clean(response)}\n{END_TOKEN}\n\n"


def take(rows, limit):
    for index, row in enumerate(rows):
        if limit >= 0 and index >= limit:
            return
        yield row


def write_pretraining(output, character_limit, seed):
    rows = dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")
    rows = rows.shuffle(seed=seed, buffer_size=10_000)
    written = 0
    next_report = 100_000_000
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            text = clean(row["text"])
            if not text:
                continue
            written += stream.write(text + f"\n{END_TOKEN}\n")
            if written >= next_report:
                print(f"FineWeb-Edu progress: {written:,} characters", flush=True)
                next_report += 100_000_000
            if written >= character_limit:
                break
    print(f"FineWeb-Edu: {written:,} characters -> {output}")


def smoltalk_records(split, limit):
    rows = dataset("HuggingFaceTB/smol-smoltalk", None, split)
    for row in take(rows, limit):
        yield from conversation_records(row["messages"])


def conversation_records(messages):
    history = []
    for message in messages:
        role = message["role"]
        content = clean(message["content"])
        if role == "assistant" and history:
            yield record(conversation_prompt(history), content)
        history.append((role, content))


def conversation_prompt(history):
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{labels.get(role, role.title())}:\n{text}" for role, text in history)


def mmlu_records(split, limit):
    rows = dataset("cais/mmlu", "all", split)
    for row in take(rows, limit):
        options = choices(row["choices"])
        answer = int(row["answer"])
        prompt = f"{row['question']}\n\n{options}"
        yield record(prompt, f"{LETTERS[answer]}. {row['choices'][answer]}")


def gsm8k_records(split, limit):
    rows = dataset("openai/gsm8k", "main", split)
    for row in take(rows, limit):
        yield record(row["question"], row["answer"])


def arc_records(split, limit):
    for subset in ("ARC-Easy", "ARC-Challenge"):
        rows = dataset("allenai/ai2_arc", subset, split)
        yield from take((arc_record(row) for row in rows), limit // 2)


def arc_record(row):
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    answer = row["answerKey"]
    index = labels.index(answer)
    return record(f"{row['question']}\n\n{options}", f"{answer}. {texts[index]}")


def choices(values):
    return "\n".join(f"{LETTERS[index]}. {value}" for index, value in enumerate(values))


def custom_records(validation=False):
    pairs = validation_pairs() if validation else training_pairs()
    return [record(instruction, response) for instruction, response in pairs]


def training_pairs():
    return [
        ("Hello", "Hello! How can I help you today?"),
        ("Hi", "Hi! What would you like help with?"),
        ("Who are you?", "I am a small language model created and trained using DeepJ."),
        ("Why is the sky blue?", "The sky looks blue because air scatters blue sunlight more strongly than red light."),
        ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid."),
        ("Spell the word necessary.", "Necessary is spelled N-E-C-E-S-S-A-R-Y."),
    ]


def validation_pairs():
    return [
        ("Good morning", "Good morning! How can I help you?"),
        ("What created you?", "I was created and trained using the DeepJ Java library."),
        ("What makes the daytime sky look blue?", "Air scatters blue sunlight more strongly than red light."),
    ]


def training_records(args, split):
    valid = split == "test"
    limit = args.validation_limit
    yield from smoltalk_records(split, limit if valid else args.smoltalk_limit)
    mmlu_split = "test" if valid else "auxiliary_train"
    mmlu_limit = limit // 2 if valid else args.mmlu_limit
    yield from mmlu_records(mmlu_split, mmlu_limit)
    yield from gsm8k_records(split, limit // 4 if valid else args.gsm8k_limit)
    yield from arc_records("test" if valid else "train", limit // 4 if valid else args.arc_limit)
    yield from custom_records(valid)


def write_records(path, records):
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for value in records:
            stream.write(value)
            count += 1
            if count % 50_000 == 0:
                print(f"Instruction progress: {count:,} records", flush=True)
    print(f"Instructions: {count:,} records -> {path}")
    return count


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_pretrain:
        write_pretraining(args.output / "fineweb-edu.txt", args.pretrain_characters, args.seed)
    if args.skip_instructions:
        return
    write_records(args.output / "instruction-train.txt", training_records(args, "train"))
    write_records(args.output / "instruction-valid.txt", training_records(args, "test"))


if __name__ == "__main__":
    prepare(arguments())
