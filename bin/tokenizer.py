import json
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--tokenizer", default="bert-base-multilingual-cased")
args = parser.parse_args()

with open(args.json) as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

new_labels = []

for text, labels in zip(data["text"], data["labels"]):
    output = tokenizer(text, return_offsets_mapping=True)
    new_labels.append([])
    i = 0
    for input_id, offsets in zip(output.input_ids, output.offset_mapping):
        while i < len(labels) and labels[i][0] < offsets[0] and labels[i][1] < offsets[1]:
            i += 1
        if i < len(labels):
            if labels[i][0] == offsets[0]:
                new_labels[-1].append("B-" + labels[i][2])
            elif labels[i][0] < offsets[0] and offsets[1] <= labels[i][1]:
                new_labels[-1].append("I-" + labels[i][2])
            else:
                new_labels[-1].append("O")
        else:
            new_labels[-1].append("O")

data["labels"] = new_labels

print(json.dumps(data, ensure_ascii=False))
