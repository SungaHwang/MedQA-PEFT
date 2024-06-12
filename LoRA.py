import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
import logging

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return x + (x @ self.A @ self.B)

class MLLoRALayer(LoRALayer):
    def forward(self, x, P):
        return x + (x @ self.A @ self.B) * (-torch.log(P))

class FisherLoRALayer(LoRALayer):
    def forward(self, x, F):
        return x + (x @ self.A @ self.B) * F

class BertWithLoRA(nn.Module):
    def __init__(self, model_name, rank, lora_type='basic', num_labels=2):
        super(BertWithLoRA, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.rank = rank
        self.lora_type = lora_type
        self.num_labels = num_labels

        if lora_type == 'basic':
            self.lora = LoRALayer(self.bert.config.hidden_size, self.bert.config.hidden_size, rank)
        elif lora_type == 'mllora':
            self.lora = MLLoRALayer(self.bert.config.hidden_size, self.bert.config.hidden_size, rank)
        elif lora_type == 'fisherlora':
            self.lora = FisherLoRALayer(self.bert.config.hidden_size, self.bert.config.hidden_size, rank)

    def forward(self, input_ids, attention_mask, labels=None, P=None, F=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        if self.lora_type == 'basic':
            sequence_output = self.lora(sequence_output)
        elif self.lora_type == 'mllora':
            if P is None:
                P = self.calculate_P(sequence_output)
            sequence_output = self.lora(sequence_output, P)
        elif self.lora_type == 'fisherlora':
            if F is None:
                F = self.calculate_F(sequence_output)
            sequence_output = self.lora(sequence_output, F)

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
        # 예측 확률 P를 계산
        with torch.no_grad():
            P = torch.softmax(sequence_output, dim=-1)
        return P

    def calculate_F(self, sequence_output):
        # 피셔 정보 행렬 F를 계산
        F = torch.var(sequence_output, dim=1, unbiased=False).unsqueeze(-1)
        F = F.expand(-1, -1, self.rank)
        return F

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
    "cola": 4,
    "mnli": 3,
    "mrpc": 4,
    "qnli": 3,
    "qqp": 3,
    "rte": 5,
    "sst2": 4,
    "stsb": 5,
    "wnli": 4,
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples, task_name):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred, task_name):
    predictions, labels = eval_pred
    if task_name == "stsb":
        return {"pearson": np.corrcoef(predictions[:, 0], labels)[0, 1]}
    else:
        preds = np.argmax(predictions, axis=1)
        if task_name in ["cola"]:
            return {"matthews_correlation": matthews_corrcoef(labels, preds)}
        else:
            return {"accuracy": (preds == labels).mean()}

all_results = {}

for lora_type in ['basic', 'mllora', 'fisherlora']:
    results = {}
    for task_name in task_to_keys.keys():
        print(f"Training for {task_name} with {lora_type}")
        datasets = load_dataset("glue", task_name)
        datasets = datasets.map(lambda examples: tokenize_function(examples, task_name), batched=True)

        if 'validation' not in datasets:
            datasets['validation'] = datasets['test']

        if task_name == "stsb":
            model = BertWithLoRA('bert-base-uncased', rank=8, lora_type=lora_type, num_labels=1).to(device)
        else:
            model = BertWithLoRA('bert-base-uncased', rank=8, lora_type=lora_type, num_labels=2).to(device)

        training_args = TrainingArguments(
            output_dir=f'./results/BERT/{task_name}_{lora_type}',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=task_to_epochs[task_name],
            weight_decay=0.01,
            logging_dir=f'./logs/BERT/{task_name}_{lora_type}',
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=task_to_best_metric[task_name],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task_name),
        )

        trainer.train()
        eval_results = trainer.evaluate()
        metric_key = task_to_best_metric[task_name]
        results[task_name] = eval_results[metric_key]
    
    all_results[lora_type] = results

df = pd.DataFrame(all_results)
df.to_csv('BERT_lora_comparison_results.csv')

print("Evaluation results saved to lora_comparison_results.csv")
