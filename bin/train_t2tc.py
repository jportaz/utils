import numpy as np
import os
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
parser.add_argument("--model", default="bert-base-multilingual-cased")
parser.add_argument("--train_dataset", required=True)
parser.add_argument("--eval_dataset")
parser.add_argument("--test_dataset")
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--max_seq_length", type=int, default=150)
parser.add_argument("--label_all_tokens", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--metric", default="seqeval", choices=["seqeval", "poseval"])
args = parser.parse_args()

#os.environ["WANDB_DISABLED"] = "true"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

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
    if args.metric == "seqeval":
        print(seqeval.metrics.classification_report([true_labels], [true_predictions], zero_division=1))
    print(sklearn.metrics.classification_report(true_labels, true_predictions, zero_division=1))
    
metric = evaluate.load(args.metric)

print(metric.inputs_description)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [ [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                         for prediction, label in zip(predictions, labels) ]

    true_labels = [ [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels) ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    if args.metric == "poseval":        
        return {
            "accuracy": results["accuracy"],
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
        }
    else:
        return {
            "accuracy": results["overall_accuracy"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

raw_datasets = load_dataset("json", data_files={"train": args.train_dataset,
                                                "valid": args.eval_dataset,
                                                "test": args.test_dataset})

label_list = get_label_list(raw_datasets["train"]["tags"])
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label_list)

print(raw_datasets)
print(raw_datasets["train"].features["tags"].feature)

# Model

model_name = args.model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=num_labels,
                                                        id2label=id2label,
                                                        label2id=label2id)

# Data sets

train_dataset = raw_datasets['train'].map(tokenize_and_align_labels, batched=True)
eval_dataset = raw_datasets['valid'].map(tokenize_and_align_labels, batched=True)
test_dataset = raw_datasets['test'].map(tokenize_and_align_labels, batched=True)

# Data collator

data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

# Initialize our Trainer

training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    fp16=args.fp16,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_f1",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
#    auto_find_batch_size=True,
    push_to_hub=False,
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# Training

checkpoint = None
#if training_args.resume_from_checkpoint is not None:
#    checkpoint = training_args.resume_from_checkpoint
#elif last_checkpoint is not None:
#    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics
trainer.save_model()  

predictions, labels, metrics = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

print_classification_report(predictions, labels)

#eval_results = trainer.evaluate()

#print(eval_results)
#print(eval_results["eval_classification_report"])

# max_train_samples = (
#     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
# )
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))

# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()

# Evaluation

# if training_args.do_eval:
#     logger.info("*** Evaluate ***")

#     metrics = trainer.evaluate()

#     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
#     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

#     trainer.log_metrics("eval", metrics)
#     trainer.save_metrics("eval", metrics)

# # Predict

# if training_args.do_predict:
#     logger.info("*** Predict ***")

#     predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
#     predictions = np.argmax(predictions, axis=2)

#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]

#     trainer.log_metrics("predict", metrics)
#     trainer.save_metrics("predict", metrics)

#     # Save predictions
#     output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
#     if trainer.is_world_process_zero():
#         with open(output_predictions_file, "w") as writer:
#             for prediction in true_predictions:
#                 writer.write(" ".join(prediction) + "\n")

# kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
# if data_args.dataset_name is not None:
#     kwargs["dataset_tags"] = data_args.dataset_name
#     if data_args.dataset_config_name is not None:
#         kwargs["dataset_args"] = data_args.dataset_config_name
#         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
#     else:
#         kwargs["dataset"] = data_args.dataset_name

# if training_args.push_to_hub:
#     trainer.push_to_hub(**kwargs)
# else:
#     trainer.create_model_card(**kwargs)
