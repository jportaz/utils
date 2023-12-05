import sys
import json

words = []
tags = []

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("#"):
        if words:
            print(json.dumps({"words": words, "tags": tags}, ensure_ascii=False))
            words = []
            tags = []
    else:
        splits = line.split("\t")
        if " " not in splits[1]:
            words.append(splits[1])        
            tags.append(splits[4][:7])
        else:
            for i, word in enumerate(splits[1].split()):
                words.append(word)                        
                tags.append(("B-" if i == 0 else "I-") + splits[4][:7])

if words:
    print(json.dumps({"words": words, "tags": tags}, ensure_ascii=False))

