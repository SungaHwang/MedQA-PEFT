from transformers import LlamaTokenizer, LlamaForCausalLM
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model = LlamaForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    target_modules=["q_proj", "v_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="Casual_LM" # set this for CLM or Seq2Seq
)

peft_model = get_peft_model(model, config)
print_trainable_parameters(peft_model)


import transformers
import datasets

finetuning_dataset_loaded = datasets.load_dataset("json", data_files="QandA.jsonl", split="train")

dataset_dict = datasets.DatasetDict({
    'train': finetuning_dataset_loaded
})

index = 200

question = dataset_dict['train'][index]['question']
answer = dataset_dict['train'][index]['answer']

prompt = f"""
Answer the following question.

{question}

Answer:
"""

batch = tokenizer(prompt, return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
print(f'\n\nMODEL GENERATION - ZERO SHOT:\n{answer}')


CUTOFF_LEN=64

def generate_prompt(data_point):
    return f"""
Answer the following question.
{data_point["question"]}

Answer:
{data_point["answer"]}

"""


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


train_data = (
    dataset_dict["train"].map(generate_and_tokenize_prompt))

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 100
OUTPUT_DIR = "experiments"

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, config)
#peft_model.print_trainable_parameters()

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir='./output',
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

for batch in train_data:
    batch["input_ids"] = batch["labels"]

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=train_data,
    eval_dataset=train_data,
    args=training_arguments,
    data_collator=data_collator
)

peft_model.config.use_cache = False

trainer.train()

peft_model.save_pretrained(OUTPUT_DIR)