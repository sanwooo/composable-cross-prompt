import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, concatenate_datasets
from trl import DataCollatorForCompletionOnlyLM

from utils.configs import DatasetConfig

class DatasetBase:
    def __init__(
            self,
            dataset_config: DatasetConfig,
    ):
        self.dataset_config = dataset_config

    def infer_unique_identifier(self) -> str:
        """
            infer the unique identifier of a dataset
        """

        unique_identifier = os.path.join(
            self.dataset_config.dataset_name,
            self.dataset_config.setting,
            re.sub(r"<blank>", str(self.dataset_config.prompt_id), "prompt_<blank>"),
        )

        return unique_identifier
    
    def _infer_dataset_path(self, split: str):
        """
            infer dataset path by concatenating unique_identifier with dataset_root
        """
        unique_identifier = self.infer_unique_identifier()
        dataset_path = os.path.join(
            self.dataset_config.dataset_root,
            unique_identifier,
            re.sub(r"<blank>", split, "<blank>.tsv"),
        )
        return dataset_path
    
    def _load_dataset(self, split: str, add_assistant=bool, reserve_columns: list[str]=list()) -> Dataset:
        """
            columns other than "messages" and reserve_columns will be dropped.
            default behavior is dropping all columns but "messages"
        """
        def _wrap_messages(example: dict, assistant: bool):
            """
                wrap user message and assistant message.
            """
            messages = list()
            messages.append({"role": "user", "content": example['user']})
            if assistant:
                messages.append({"role": "assistant", "content": str(example['assistant'])})
            
            return {'messages': messages}

        dataset_path = self._infer_dataset_path(split)
        dataset = Dataset.from_csv(dataset_path, sep='\t')
        return dataset.map(
            _wrap_messages, fn_kwargs=dict(assistant=add_assistant),
            remove_columns=set(dataset.column_names) - set(reserve_columns), 
            load_from_cache_file=False,
        )
                    
    def _tokenize(
            self,   
            examples: dict,
            add_generation_prompt: bool,
            tokenizer: AutoTokenizer,
    ):
        chat_batch = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # remove <think token> for deepseek-r1 model
        chat_batch = [re.sub(r"<think>\n", "", x) for x in chat_batch]
        inputs = tokenizer(chat_batch, add_special_tokens=False, truncation=True)
        return inputs
    
    def get_data_collator(self, tokenizer: AutoTokenizer, response_template: list[int]=None) -> DataCollatorForCompletionOnlyLM | DataCollatorWithPadding:
        if response_template:
            # in a single-turn case, we don't need to pass the instruction template (<|start_header_id|>user<|end_header_id|>\n\n for the case of Llama3-8B-Instruct)
            data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, \
                response_template=response_template, instruction_template=None, mlm=False)
            return data_collator
        # else:
            # data_collator = DataCollatorWithPadding(tokenizer)
        else:
            def collate_fn(batch: dict[str, list[int]]):
                data_collator = DataCollatorWithPadding(tokenizer)
                pad_batch = data_collator(batch)
                # pad_batch["labels"] = (pad_batch["score"] - pad_batch["min_score"]).long()
                return pad_batch
            return collate_fn

    def load_dataset(self):
        pass

class SFTDataset(DatasetBase):
    def __init__(
            self,
            dataset_config: DatasetConfig,
    ):
        super().__init__(dataset_config)

    def load_dataset(self, reserve_columns: list[str]=list()):
        split_list = ['train', 'dev']
        return DatasetDict({
            split: self._load_dataset(split, add_assistant=True, reserve_columns=reserve_columns) \
            for split in split_list
        })
    
    def load_data_loader(
            self,
            tokenizer: AutoTokenizer,
            batch_size: int,
            response_template: list[int] = None,
        ) -> tuple[DataLoader, DataLoader]:
        reserve_columns = ['essay_id', 'score', 'min_score', 'max_score']
        dataset_dict = self.load_dataset(reserve_columns=reserve_columns)
        data_collator = self.get_data_collator(tokenizer, response_template)

        # align the token corresponding to the score to the rightmost index
        tokenizer.padding_side = 'left' 
        data_loader_dict = dict()
        shuffle_map = {'train': True, 'dev': False}
        for split in ['train', 'dev']:
            dataset = dataset_dict[split]
            tokenized_dataset = dataset.map(
                self._tokenize, fn_kwargs=dict(add_generation_prompt=False, tokenizer=tokenizer),
                remove_columns=set(dataset.column_names) - set(reserve_columns),
                batched=True, load_from_cache_file=False,
            )
            data_loader = DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                shuffle=shuffle_map[split], 
                collate_fn=data_collator,
            )
            data_loader_dict[split] = data_loader
        
        return data_loader_dict['train'], data_loader_dict['dev']


class EvalDataset(DatasetBase):
    def __init__(
            self,
            dataset_config: DatasetConfig,
    ):
        super().__init__(dataset_config)
    
    def load_dataset(self, reserve_columns: list[str]=list()) -> Dataset:
        return self._load_dataset(split=self.dataset_config.split, add_assistant=False, reserve_columns=reserve_columns)

    def load_data_loader(
            self,
            tokenizer: AutoTokenizer,
            batch_size: int,
            response_template: list[int] = None,
            shuffle=False,
            n_sample: int=None,
    ) -> DataLoader:
        reserve_columns = ['essay_id', 'score', 'min_score', 'max_score']
        dataset = self.load_dataset(reserve_columns=reserve_columns)
        data_collator = self.get_data_collator(tokenizer, response_template)

        # align the token corresponding to the score to the rightmost index
        tokenizer.padding_side = 'left' 
        tokenized_dataset = dataset.map(
            self._tokenize, fn_kwargs=dict(add_generation_prompt=True, tokenizer=tokenizer),
            remove_columns=set(dataset.column_names) - set(reserve_columns),
            batched=True, load_from_cache_file=False,
        )
        if n_sample:
            tokenized_dataset = tokenized_dataset.select(
                indices=np.random.choice(a=len(tokenized_dataset), size=n_sample, replace=False),
            )
        data_loader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
        )
        return data_loader
    
