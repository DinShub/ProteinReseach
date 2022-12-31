import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np


# The modal we are using
model_checkpoint = "facebook/esm2_t12_35M_UR50D"

dataset_all = pd.read_csv('./all.csv')

train_seq, test_seq, train_labels, test_labels = train_test_split(dataset_all['sequence'].values, dataset_all['label'].values, test_size=0.3, shuffle=True)

# train_seq, valid_seq, train_labels, valid_labels = train_test_split(pre_train_seq, pre_train_labels, test_size=0.2, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


train_tokenized = tokenizer(train_seq.tolist())
test_tokenized = tokenizer(test_seq.tolist())

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column('labels', train_labels)
test_dataset = test_dataset.add_column('labels', test_labels)

num_labels = max(train_labels.tolist() + test_labels.tolist()) + 1  # Add 1 since 0 can be a label
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

model_name = model_checkpoint.split("/")[-1]
batch_size = 8

args = TrainingArguments(
  f"{model_name}-finetuned-localization",
  evaluation_strategy = "epoch",
  save_strategy = "epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  num_train_epochs=3,
  weight_decay=0.01,
  load_best_model_at_end=True,
  metric_for_best_model="accuracy",
)

metric = load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return metric.compute(predictions=predictions, references=labels)
  
trainer = Trainer(
  model,
  args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics,
)

trainer.train()