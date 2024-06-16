import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class EnhancedLoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, info_type='none'):
        super(EnhancedLoRALayer, self).__init__()
        self.rank = rank
        self.info_type = info_type
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))

    def forward(self, x, info=None):
        if self.info_type == 'mllora' and info is not None:
            return x + (x @ self.W) * (-torch.log(info))
        elif self.info_type == 'fisherlora' and info is not None:
            info_expanded = info.unsqueeze(1).expand_as(x @ self.W)
            return x + (x @ self.W) * info_expanded
        else:
            return x + x @ self.W

class BertWithEnhancedLoRA(nn.Module):
    def __init__(self, model_name, rank, info_type='none', num_labels=2):
        super(BertWithEnhancedLoRA, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.rank = rank
        self.info_type = info_type
        self.num_labels = num_labels
        self.lora = EnhancedLoRALayer(self.bert.config.hidden_size, self.bert.config.hidden_size, rank, info_type)

    def forward(self, input_ids, attention_mask, labels=None, info=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.info_type in ['mllora', 'fisherlora'] and info is None:
            if self.info_type == 'mllora':
                info = self.calculate_P(sequence_output)
            elif self.info_type == 'fisherlora':
                info = self.calculate_F(sequence_output)

        sequence_output = self.lora(sequence_output, info)
        logits = self.bert.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

    def calculate_P(self, sequence_output):
        with torch.no_grad():
            P = torch.softmax(sequence_output, dim=-1)
        return P

    def calculate_F(self, sequence_output):
        F = torch.var(sequence_output, dim=1, unbiased=False).mean(dim=-1).unsqueeze(1)
        return F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_best_metric = {
    "rte": "eval_accuracy",
    "mrpc": "eval_f1", 
    "cola": "eval_matthews_correlation", 
    "stsb": "eval_pearson", 
    "sst2": "eval_accuracy", 
    "qnli": "eval_accuracy",
    "mnli": "eval_accuracy",
    "wnli": "eval_accuracy",
    "qqp": "eval_accuracy",
}

task_to_epochs = {
    "cola": 15,
    "mnli": 10,
    "mrpc": 15,
    "qnli": 10,
    "qqp": 10,
    "rte": 15,
    "sst2": 15,
    "stsb": 15,
    "wnli": 15,
}

task_to_validation_split = {
    "mnli": "validation_matched"
}

def tokenize_function(examples, task_name):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred, task_name):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    if task_name == "stsb":
        return {"pearson": np.corrcoef(predictions[:, 0], labels)[0, 1]}
    elif task_name == "cola":
        return {"matthews_correlation": matthews_corrcoef(labels, preds)}
    elif task_name == "mrpc":
        f1 = f1_score(labels, preds)
        return {"f1": f1}
    else:
        return {"accuracy": (preds == labels).mean()}

for rank in [8, 6, 4, 10]:
    all_results = {}
    for lora_type in ['basic', 'mllora', 'fisherlora']:
        results = {}
        for task_name in task_to_keys.keys():
            print(f"Training for {task_name} with {lora_type} and rank {rank}")
            try:
                datasets = load_dataset("glue", task_name)

                validation_split = task_to_validation_split.get(task_name, "validation")
                if validation_split not in datasets:
                    print(f"Validation split '{validation_split}' not found for task '{task_name}', skipping.")
                    continue

                if len(datasets["train"]) > 2000:
                    train_dataset = datasets["train"].shuffle(seed=42).select(range(2000))
                else:
                    train_dataset = datasets["train"]

                if len(datasets[validation_split]) > 2000:
                    validation_dataset = datasets[validation_split].shuffle(seed=42).select(range(2000))
                else:
                    validation_dataset = datasets[validation_split]

                tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, task_name), batched=True)
                tokenized_validation_dataset = validation_dataset.map(lambda examples: tokenize_function(examples, task_name), batched=True)

                if task_name == "stsb":
                    num_labels = 1
                elif task_name == "mnli":
                    num_labels = 3
                elif task_name in ["mrpc", "qqp", "qnli", "rte", "wnli", "sst2", "cola"]:
                    num_labels = 2
                else:
                    num_labels = 2

                model = BertWithEnhancedLoRA('bert-base-uncased', rank=rank, info_type=lora_type, num_labels=num_labels).to(device)

                training_args = TrainingArguments(
                    output_dir=f'./results/BERT2/{task_name}_{lora_type}_rank{rank}',
                    evaluation_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    num_train_epochs=task_to_epochs[task_name],
                    weight_decay=0.01,
                    logging_dir=f'./logs/BERT2/{task_name}_{lora_type}_rank{rank}',
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model=task_to_best_metric[task_name],
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    eval_dataset=tokenized_validation_dataset,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task_name),
                )

                trainer.train()
                eval_results = trainer.evaluate()
                metric_key = task_to_best_metric[task_name]
                results[task_name] = eval_results[metric_key]
            except Exception as e:
                print(f"Error occurred while training {task_name} with {lora_type} and rank {rank}: {str(e)}")

        all_results[f"{lora_type}"] = results

    df = pd.DataFrame(all_results)
    df.to_csv(f'BERT2_lora_comparison_results_rank{rank}.csv')

print("Evaluation results saved to separate CSV files for each rank")
