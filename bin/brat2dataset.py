# Assumes:
#  (1) continuity (no ";" in the offset field)
#  (2) no label overlapping

import argparse
import json
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--ann", required=True)
parser.add_argument("--txt", required=True)
args = parser.parse_args()

with open(args.ann, "r") as f:
    ann = f.readlines()
    ann = [line.split("\t")[1].split(" ") for line in ann]
    ann = [(int(start), int(end), label) for label, start, end in ann]

dataset = { "data": {"examples": []} }

with open(args.txt, "r") as f:
    dataset["text"] = f.readlines()
    dataset["offsets"] = []
    dataset["labels"] = []
    start = 0
    i = 0
    for text in dataset["text"]:
        dataset["labels"].append([])
        end = start + len(text)
        dataset["offsets"].append((start, end))
        while i < len(ann) and start <= ann[i][0] and ann[i][1] <= end:
            dataset["labels"][-1].append((ann[i][0] - start, ann[i][1] - start, ann[i][2]))
            i += 1
        start += len(text)

dataset["labels"]  = [ann for text, ann in zip(dataset["text"], dataset["labels"]) if text != "\n"]
dataset["offsets"] = [offsets for text, offsets in zip(dataset["text"], dataset["offsets"]) if text != "\n"]
dataset["text"] = [text for text in dataset["text"] if text != "\n"]

print(json.dumps(dataset, ensure_ascii=False))
