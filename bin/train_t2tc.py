from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
import evaluate

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

padding = False #"max_length" if data_args.pad_to_max_length else False
max_seq_length = 512
label_all_tokens = False

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
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
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)


    # Remove ignored index (special tokens)
    true_predictions = [ [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                         for prediction, label in zip(predictions, labels) ]

    true_labels = [ [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels) ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # if data_args.return_entity_level_metrics:
    #     # Unpack nested dictionaries
    #     final_results = {}
    #     for key, value in results.items():
    #         if isinstance(value, dict):
    #             for n, v in value.items():
    #                 final_results[f"{key}_{n}"] = v
    #         else:
    #             final_results[key] = value
    #     return final_results
    # else:
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

model_name = 'bert-base-multilingual-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

raw_datasets = load_dataset("json", data_files="/tmp/jj")

label_list = get_label_list(raw_datasets["train"]["tags"])
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

print(label_list)
print(raw_datasets)
print(raw_datasets["train"].features["tags"].feature)

#print(tokenize_and_align_labels([dataset["train"][0]]))

train_dataset = raw_datasets['train'].map(tokenize_and_align_labels, batched=True)
eval_dataset = train_dataset

print(train_dataset[1])

print(train_dataset)

# Data collator

data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8) # if training_args.fp16 else None)

# Initialize our Trainer

trainer = Trainer(
    model=model,
    #args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training

checkpoint = None
#if training_args.resume_from_checkpoint is not None:
#    checkpoint = training_args.resume_from_checkpoint
#elif last_checkpoint is not None:
#    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics
trainer.save_model()  # Saves the tokenizer too for easy upload

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
