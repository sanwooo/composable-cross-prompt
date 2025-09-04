import os, re
import torch
from peft import PeftModel, LoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format
from utils.logger import logger
from utils.constants import RESERVED_TOKEN_MAPPING, RESPONSE_TEMPLATE_MAPPING

def setup_for_instruction_tuning(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_name: str):

    tokenizer.padding_side = 'right' # this is recommended by SFTTrainer
    tokenizer.truncation_side = 'left' # leftmost content (system prompt) is less important than rightmost content (assistant message, e.g. essay score)
    # below is in accordance to the following error: You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. 
    if model_name in  ['Phi-4-mini-instruct']:
        tokenizer.padding_side = 'left'
    
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        logger.info("pad_token_id is not defined or pad_token_id is equal to eos_token_id. specifying a pad_token...")
        # use reserved token instead without adding new token and resize model embedding
        tokenizer.pad_token = RESERVED_TOKEN_MAPPING[model_name] 
        
    logger.info(f"tokenizer special token: {tokenizer.special_tokens_map}")

    response_template = RESPONSE_TEMPLATE_MAPPING[model_name]
    return model, tokenizer, response_template


def load_peft_ckpt(model_dir: str, model_name: str, ckpt_path: str):
    # sanity check
    assert re.search(f'/{model_name}/', ckpt_path) is not None, "fine-tuned checkpoint model does not match pre-trained model"

    # load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_dir, model_name), 
            attn_implementation='flash_attention_2',
            **dict(torch_dtype=torch.bfloat16),
        ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))
    # setup for instruction tuning format
    # In particular, 1. add special token to tokenizer 2. resize model token embedding
    model, tokenizer, response_template = setup_for_instruction_tuning(model, tokenizer, model_name)
    # load peft component
    peft_model = PeftModel.from_pretrained(model, ckpt_path)
    peft_model = peft_model.merge_and_unload()
    return peft_model, tokenizer, response_template

def load_adapters(
        model_dir: str,
        model_name: str,
        ckpt_root: str,
        merge_prompt_ids: list[str],
        seed: int,
) -> tuple[LoraModel, AutoTokenizer, list[int]]:
    # sanity check
    assert re.search(f'/{model_name}/', ckpt_root) is not None, "fine-tuned checkpoint model does not match pre-trained model"

    # load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_dir, model_name), 
            # attn_implementation='flash_attention_2',
            **dict(torch_dtype=torch.bfloat16, output_hidden_states=True),
        ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))
    # setup for instruction tuning format
    # In particular, 1. add special token to tokenizer 2. resize model token embedding
    model, tokenizer, response_template = setup_for_instruction_tuning(model, tokenizer, model_name)
    # load peft adapters
    def apply_adapter_name_template(prompt_id: int):
        adapter_name_template = r"prompt_<blank>"
        return re.sub('<blank>', str(prompt_id), adapter_name_template)
    
    def apply_seed_template(seed_: int):
        seed_template = r"seed_<blank>"
        return re.sub('<blank>', str(seed_), seed_template)
    
    model: LoraModel = PeftModel.from_pretrained(model, os.path.join(ckpt_root, apply_adapter_name_template(merge_prompt_ids[0]), apply_seed_template(seed)),\
                                      adapter_name=apply_adapter_name_template(merge_prompt_ids[0]))
            
    for i in range(1, len(merge_prompt_ids)):
        model.load_adapter(os.path.join(ckpt_root, apply_adapter_name_template(merge_prompt_ids[i]), apply_seed_template(seed)), \
                           adapter_name=apply_adapter_name_template(merge_prompt_ids[i]))

    # model.to(torch.bfloat16)
    return model, tokenizer, response_template


