import numpy as np
import os
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from datasets import load_dataset
import sklearn
import seqeval
from seqeval import metrics
import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoints")
parser.add_argument("--test_dataset", required=True)
parser.add_argument("--label_all_tokens", action="store_true")
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--metric", default="seqeval", choices=["seqeval", "poseval"])
args = parser.parse_args()

#os.environ["WANDB_DISABLED"] = "true"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

padding = False #"max_length" if data_args.pad_to_max_length else False

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"],
                                 padding=padding,
                                 truncation=True,
                                 max_length=args.max_seq_length,
                                 is_split_into_words=True)

    labels = []

    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if args.label_all_tokens:
                    label_ids.append(b_to_i_label[label2id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Metrics

def print_classification_report(predictions, labels):
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_predictions.append(id2label[p])
                true_labels.append(id2label[l])
    #print(seqeval.metrics.classification_report([true_labels], [true_predictions], zero_division=1))
    print(sklearn.metrics.classification_report(true_labels, true_predictions, zero_division=1))
    
raw_datasets = load_dataset("json", data_files={"test": args.test_dataset})

# Model

config = AutoConfig.from_pretrained(args.model)

#add_prefix_space = config.model_type in {'gpt2', 'roberta'}
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True) #, add_prefix_space=add_prefix_space)

label2id = config.label2id
id2label = config.id2label

model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(model=model, data_collator=data_collator)

#tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, num_proc=5, batch_size=1000)
test_dataset = raw_datasets['test'].map(tokenize_and_align_labels, batched=True)

predictions, labels, metrics = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

print_classification_report(predictions, labels)

print("pred", "label", sep="\t")

for data, prediction, label in zip(test_dataset, predictions, labels):
    it = iter(data["words"])
    for (p, l) in zip(prediction, label):
        if l != -100:
            print(p == l, id2label[p], id2label[l], next(it), sep="\t")
    print()

