import sys
import json
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from handler import DataHandler

assert torch.cuda.is_available(), "No cuda device detected"


class Inferer:
    """
    A basic inference class for accessing medAlpaca models programmatically.

    This class provides methods for loading supported medAlpaca models, tokenizing inputs,
    and generating outputs based on the specified model and configurations.

    Attributes:
        available_models (dict): A dictionary containing the supported models and their configurations.

    Args:
        model_name (str): The name of the medAlpaca model to use for inference.
        prompt_template (str): The path to the JSON file containing the prompt template.
        base_model (str, optional): If LoRA is used, this should point to the bases model weigts
        model_max_length: (int, optional): Number of input tokens to the model. Default is 512.
        load_in_8bit (bool, optional): Wether a quantized model should be loaded. Default is False
        torch_dtype (torch.dtype, optional): The torch datatype to load the base model. Default is float16
        peft (bool, optional): If the model was trainied in 8bit or with LoRA, PEFT library should be used
            to load the model. Default is False. 

    Example:

        medalpaca = medAlapaca("medalpaca/medalapca-7b", "prompts/alpaca.json")
        response = medalpaca(input="What is Amoxicillin?")
    """
        
    def __init__(
        self, 
        model_name: str, 
        prompt_template: str,
        base_model: str = None,
        model_max_length: int = 512,
        load_in_8bit: bool = False, 
        torch_dtype: torch.dtype = torch.float16, 
        peft: bool = False
    ) -> None:
        
        if base_model and not peft: 
            raise ValueError(
                "You have specified a base model, but `peft` is false", 
                "This would load the base model only"
            )
            
        self.model = self._load_model(
            model_name = model_name, 
            base_model = base_model or model_name, 
            load_in_8bit = load_in_8bit, 
            torch_dtype = torch_dtype, 
            peft = peft
        )
        
        tokenizer = self._load_tokenizer(base_model or model_name)
                
        self.data_handler = DataHandler(
            tokenizer,
            prompt_template = prompt_template, 
            model_max_length = model_max_length,
            train_on_inputs = False,
        )
        
   
    def _load_model(
        self, 
        model_name: str, 
        base_model: str, 
        load_in_8bit: bool, 
        torch_dtype: torch.dtype, 
        peft: bool
    ) -> torch.nn.Module:

        if "llama" in base_model.lower(): 
            load_model = LlamaForCausalLM
        else: 
            load_model = AutoModelForCausalLM
            
        model = load_model.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            device_map={"": 0},
        )

        if peft:
            model = PeftModel.from_pretrained(
                model,
                model_id=model_name,
                torch_dtype=torch_dtype,
                device_map={"": 0},
            )
        
        if not load_in_8bit:
            model.half()
            
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        return model
    
    def _load_tokenizer(self, model_name: str): 
        if "llama" in model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"    
        return tokenizer
        
    def __call__(
        self,
        input: str,
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from the medAlpaca model using the given input and instruction.

        Args:
            input (str):
                The input text to provide to the model.
            instruction (str, optional):
                An optional instruction to guide the model's response.
            output (str, optional): 
                Prepended to the models output, e.g. for 1-shot prompting
            max_new_tokens (int, optional): 
                How many new tokens the model can generate
            verbose (bool, optional): 
                If True, print the prompt before generating a response.
            **generation_kwargs:
                Keyword arguments to passed to the `GenerationConfig`.
                See here for possible arguments: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation

        Returns:
            str: The generated response from the medAlpaca model.
        """

        prompt = self.data_handler.generate_prompt(instruction = instruction, input = input, output = output)
        if verbose:
            print(prompt)
            
        input_tokens = self.data_handler.tokenizer(prompt, return_tensors="pt")
        input_token_ids = input_tokens["input_ids"].to("cuda")

        generation_config = GenerationConfig(**generation_kwargs)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        generation_output_decoded = self.data_handler.tokenizer.decode(generation_output.sequences[0])
        split = f'{self.data_handler.prompt_template["output"]}{output or ""}'
        response = generation_output_decoded.split(split)[-1].strip()
        return response

"""Run evaluation of medAlpaca models on the USMLE self assessment. 
The questions can be downloaded at: https://huggingface.co/medalapca/

Example evaluation: 

Assume you have downloaded the steps into the folder "usmle" you can now evaluate 
a medalpaca model: 

```bash
export HF_HOME=/path/to/hf_cache

python eval_usmle.py \
    --model_name 'medalpaca/medalpaca-lora-7b-8bit' \
    --prompt_template '../medalpaca/prompt_templates/medalpaca.json' \
    --base_model 'decapoda-research/llama-13b-hf' \
    --peft True \
    --load_in_8bit True \
    --path_to_exams 'data/test/'

This will create three new files in 'data/test', named stepX_MODELNAME.json. 

The generation methods it hardcoded to the `sampling` dict, feel free to adapt this

"""

import sys
import os
sys.path.append("..")

import re
import json
import fire
import string

from tqdm.autonotebook import tqdm
#from medalpaca.inferer import Inferer

greedy_search = {
    "num_beams" : 1, 
    "do_sample" : False,
    "max_new_tokens" : 128, 
    "early_stopping" : False
}

beam_serach = {
    "num_beams" : 4, 
    "do_sample" : False,
    "max_new_tokens" : 128, 
    "early_stopping" : True,
}

sampling_top_k = {
    "do_sample" : True,
    "num_beams": 1,
    "max_new_tokens": 128, 
    "early_stopping": True,
    "temperature": 0.7,
    "top_k": 50
}

sampling_top_p = {
    "do_sample" : True,
    "top_k": 0, 
    "num_beams": 1,
    "max_new_tokens": 128, 
    "early_stopping": True,
    "temperature": 0.7,
    "top_p": 0.9
}

sampling = {
    "do_sample" : True,
    "top_k": 50, 
    "num_beams": 1,
    "max_new_tokens": 128, 
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9
}


def format_question(d): 
    question = d["question"]
    options = d["options"]
    for k, v in options.items(): 
        question += f"\n{k}: {v}"
    return question


def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str
    
    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""

def starts_with_capital_letter(input_str):
    """
    The answers should start like this: 
        'A: '
        'A. '
        'A '
    """
    pattern = r'^[A-Z](:|\.|) .+'
    return bool(re.match(pattern, input_str))


def main(
    model_name: str, # "medalpaca/medalpaca-lora-13b-8bit", 
    prompt_template: str, #"medalpaca\prompt_templates\medalpaca.json"
    base_model: str, # "decapoda-research/llama-13b-hf",
    peft: bool, # True,
    load_in_8bit: bool, # True
    path_to_exams: str, # eval/data/test/
    ntries: int = 5, 
    skip_if_exists: bool = True,
):
    
    model = Inferer(
        model_name=model_name,
        prompt_template=prompt_template,
        base_model=base_model,
        peft=peft,
        load_in_8bit=load_in_8bit,
    ) 
    
    for step_idx in [1,2,3]: 
        
        with open(os.path.join(path_to_exams, f"step{step_idx}.json")) as fp: 
            step = json.load(fp)   
        outname = os.path.join(path_to_exams, f"step{step_idx}_{model_name.split('/')[-1]}.json")
        if os.path.exists(outname): 
            with open(outname, "r") as fp:
                answers = json.load(fp)
        else: 
            answers = []
        
        pbar = tqdm(step)
        pbar.set_description_str(f"Evaluating USMLE Step {step_idx}")
        for i, question in enumerate(pbar): 
            if skip_if_exists and (i+1) <= len(answers):
                continue
            for j in range(ntries): 
                response = model(
                    instruction="Answer this multiple choice question.", 
                    input=format_question(question), 
                    output="The Answer to the question is:",
                    **sampling
                )
                response = strip_special_chars(response)
                if starts_with_capital_letter(response): 
                    pbar.set_postfix_str(f"")
                    break
                else: 
                    pbar.set_postfix_str(f"Output not satisfactoy, retrying {j+1}/{ntries}")
            question["answer"] = response
            answers.append(question)
            with open(outname, "w+") as fp:
                json.dump(answers, fp)


if __name__ == "__main__": 
    fire.Fire(main)
