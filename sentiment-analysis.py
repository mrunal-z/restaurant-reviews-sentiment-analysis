# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import nltk
import sklearn

!pip install datasets transformers huggingface_hub sentencepiece

import torch
from datasets import Dataset, DatasetDict
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from transformers import RobertaTokenizer, BertTokenizer
from transformers import XLNetTokenizer
from transformers import LukeTokenizer
import os

os.environ["WANDB_DISABLED"] = "true"

"""### Importing and splitting the dataset"""

df=pd.read_excel('/content/restaurant_reviews.xlsx')
df.head()

X = df.iloc[:,0]
Y = df.iloc[:,1]

Xtrain, Xtest, Ytrain, Ytest = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state = 1, stratify=Y)
Xtrain, Xval, Ytrain, Yval = sklearn.model_selection.train_test_split(Xtrain, Ytrain, test_size=0.2, random_state = 1, stratify=Ytrain)

train_data = Dataset.from_pandas(pd.concat([Xtrain, Ytrain], axis=1), preserve_index=False)
test_data = Dataset.from_pandas(pd.concat([Xtest, Ytest], axis=1), preserve_index=False)
val_data = Dataset.from_pandas(pd.concat([Xval, Yval], axis=1), preserve_index=False)

dataset = DatasetDict({
    'train': train_data,
    'test': test_data,
    'val': val_data})

dataset

"""## Model 1: RoBERTa

### Tokenizing
"""

tokenizer_rob = RobertaTokenizer.from_pretrained("roberta-base")
data_collator_rob = DataCollatorWithPadding(tokenizer=tokenizer_rob)

def tokenize_function_rob(examples):
  return tokenizer_rob(examples["review"], truncation = True, max_length = 512, padding = True)

tokenized_datasets_rob = dataset.map(tokenize_function_rob, batched=True)

"""### Training"""

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", num_train_epochs=15)   # default arguments for fine-tuning

def compute_metrics(eval_preds):   # compute accuracy and f1-score
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

.model_rob = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)  # overwriting MLM roberta-base for sequence binary classification

trainer_rob = Trainer(   # specifying trainer class
    model_rob,
    training_args,
    train_dataset=tokenized_datasets_rob['train'],
    eval_dataset=tokenized_datasets_rob['val'],
    data_collator=data_collator_rob,
    tokenizer=tokenizer_rob,
    compute_metrics=compute_metrics,
)

trainer_rob.train()  # starts fine-tuning

"""### Testing"""

predictions_rob = trainer_rob.predict(tokenized_datasets_rob['test'])
predictions_rob[2]

"""## Model 2: XLNet

### Tokenizing
"""

tokenizer_xl = AutoTokenizer.from_pretrained("xlnet-base-cased")
tokenizer_xl.pad_token = tokenizer_xl.eos_token

data_collator_xl = DataCollatorWithPadding(tokenizer=tokenizer_xl)

def tokenize_function_xl(examples):
  return tokenizer_xl(examples["review"], truncation = True, max_length = 512, padding = True)

tokenized_datasets_xl = dataset.map(tokenize_function_xl, batched=True)

"""### Training"""

model_xl = AutoModelForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)  # overwriting MLM xlnet-base-cased for sequence binary classification

trainer_xl = Trainer(   # specifying trainer class
    model_xl,
    training_args,
    train_dataset=tokenized_datasets_xl['train'],
    eval_dataset=tokenized_datasets_xl['val'],
    data_collator=data_collator_xl,
    tokenizer=tokenizer_xl,
    compute_metrics=compute_metrics,
)

trainer_xl.train()  # starts fine-tuning

"""### Testing"""

predictions_xl = trainer_xl.predict(tokenized_datasets_xl['test'])
predictions_xl[2]

"""## Model 3: LUKE

### Tokenizing
"""

tokenizer_luke = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
tokenizer_luke.pad_token = tokenizer_luke.eos_token

data_collator_luke = DataCollatorWithPadding(tokenizer=tokenizer_luke)

def tokenize_function_luke(examples):
  return tokenizer_luke(examples["review"], truncation = True, max_length = 512, padding = True)

tokenized_datasets_luke = dataset.map(tokenize_function_luke, batched=True)

"""### Training

"""

model_luke = AutoModelForSequenceClassification.from_pretrained("studio-ousia/luke-base", num_labels=2)  # overwriting MLM "studio-ousia/luke-base" for sequence binary classification

trainer_luke = Trainer(   # specifying trainer class
    model_luke,
    training_args,
    train_dataset=tokenized_datasets_luke['train'],
    eval_dataset=tokenized_datasets_luke['val'],
    data_collator=data_collator_luke,
    tokenizer=tokenizer_luke,
    compute_metrics=compute_metrics,
)

trainer_luke.train()  # starts fine-tuning

"""### Testing"""

predictions_luke = trainer_luke.predict(tokenized_datasets_luke['test'])
predictions_luke[2]

"""## Model 4: BERT

### Tokenizing
"""

tokenizer_bert = BertTokenizer.from_pretrained("bert-base-cased")
data_collator_bert = DataCollatorWithPadding(tokenizer=tokenizer_bert)

def tokenize_function_bert(examples):
  return tokenizer_bert(examples["review"], truncation = True, max_length = 512, padding = True)

tokenized_datasets_bert = dataset.map(tokenize_function_bert, batched=True)

"""### Training"""

model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)  # overwriting MLM "bert-base-uncased" for sequence binary classification

trainer_bert = Trainer(   # specifying trainer class
    model_bert,
    training_args,
    train_dataset=tokenized_datasets_bert['train'],
    eval_dataset=tokenized_datasets_bert['val'],
    data_collator=data_collator_bert,
    tokenizer=tokenizer_bert,
    compute_metrics=compute_metrics,
)

trainer_bert.train()  # starts fine-tuning

"""### Testing"""

predictions_bert = trainer_bert.predict(tokenized_datasets_bert['test'])
predictions_bert[2]

"""## Model 5: GPT

### Tokenizing
"""

model_name = "ydshieh/tiny-random-gptj-for-sequence-classification"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
  return tokenizer(examples["review"], truncation = True, max_length = 512, padding = True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

"""### Training"""

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

metric = load_metric("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()

"""### Testing"""

predictions_gpt = trainer.predict(tokenized_datasets['test'])
predictions_gpt[2]

"""## Plotting graphs"""

import matplotlib.pyplot as plt

#plotting accuracies
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X = np.arange(5)
models = ['RoBERTa','XLNet' , 'LUKE', 'BERT', 'GPT']
values = [[0.904762, 0.928571,0.952381,0.904762, 0.865],[0.9423076923076923,0.9807692307692307,0.9807692307692307,0.9230769230769231, 0.75]]
ax.bar(X + 0.00, values[0], color = 'r', width = 0.25, label = 'Train Accuracy')
ax.bar(X + 0.25, values[1], color = 'c', width = 0.25, label = 'Test Accuracy')
ax.set_xticks(X)
ax.set_xticklabels(models)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#plotting f1 scores
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X = np.arange(5)
models = ['RoBERTa','XLNet' , 'LUKE', 'BERT', 'GPT']
values = [[0.916667,0.936170,0.956522,0.916667, 0.86],[0.9473684210526316,0.9818181818181818,0.9818181818181818,0.9285714285714286, 0.75]]
ax.bar(X + 0.00, values[0], color = 'r', width = 0.25, label = 'Train F1')
ax.bar(X + 0.25, values[1], color = 'c', width = 0.25, label = 'Test F1')
ax.set_xticks(X)
ax.set_xticklabels(models)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()