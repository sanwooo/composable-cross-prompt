import os
import argparse
from dataclasses import asdict
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from utils.logger import logger
from data.processor import DatasetBase, SFTDataset

from model.utils import setup_for_instruction_tuning
from utils.utils import create_experiment_directory, update_dataclass_with_args, delete_subdirs_with_regex
from utils.configs import DatasetConfig, CustomLoraConfig

logging.basicConfig(level=logging.INFO)

def setup_experiment():
    """
        read out arguments.
    """
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--experiment_name", type=str, default=None)
    cmd_parser.add_argument("--seed", type=int, default=42)
    cmd_parser.add_argument("--dataset_name", type=str, choices=["ASAP", "PERSUADE2.0"])
    cmd_parser.add_argument("--setting", type=str, choices=['cross-prompt', 'cross-prompt-individual'])
    cmd_parser.add_argument("--prompt_id", type=int)
    cmd_parser.add_argument("--model_dir", type=str)
    cmd_parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Phi-4-mini-instruct"])
    cmd_parser.add_argument("--config_path", type=str)
    cmd_parser.add_argument("--early_stopping_patience", type=int, default=3)
    cmd_args = cmd_parser.parse_args()

    # create experiment directory
    experiment_dir = create_experiment_directory(
        project_root=".", 
        file_name=os.path.splitext(os.path.basename(__file__))[0],
        experiment_name=cmd_args.experiment_name,
    )

    # update configs
    parser = HfArgumentParser((SFTConfig, CustomLoraConfig, DatasetConfig))
    sft_config, custom_lora_config, dataset_config = parser.parse_yaml_file(cmd_args.config_path)
    lora_config = LoraConfig(**asdict(custom_lora_config))
    update_dataclass_with_args(dataset_config, cmd_args)
    update_dataclass_with_args(sft_config, cmd_args)
    
    # update checkpoint directory (output_dir)
    dataset_identifier = DatasetBase(dataset_config=dataset_config).infer_unique_identifier()
    sft_config.output_dir = os.path.join(
        experiment_dir,
        sft_config.output_dir,
        cmd_args.model_name,
        dataset_identifier,
        f"seed_{sft_config.seed}",
    )

    logger.info(f"cmd_args: {vars(cmd_args)}\n")
    logger.info(f"sft_config: {vars(sft_config)}\n")
    logger.info(f"lora_config: {vars(lora_config)}\n")
    logger.info(f"dataset_config: {vars(dataset_config)}\n")

    return cmd_args, sft_config, lora_config, dataset_config

if __name__ == "__main__":

    args, sft_config, lora_config, dataset_config = setup_experiment()
    model_path = os.path.join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            # attn_implementation='flash_attention_2',
            **dict(torch_dtype=torch.bfloat16),
        ).to('cuda')
        
    model, tokenizer, response_template = setup_for_instruction_tuning(model, tokenizer, args.model_name)

    dataset_reader = SFTDataset(
        dataset_config=dataset_config,
    )
    dataset = dataset_reader.load_dataset()
    data_collator = dataset_reader.get_data_collator(tokenizer, response_template)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        args=sft_config,
        peft_config=lora_config,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()
    delete_subdirs_with_regex(sft_config.output_dir, r"checkpoint-\d+")
