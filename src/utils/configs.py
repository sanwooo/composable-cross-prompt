from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    dataset_root: str = "./assets"
    dataset_name: str = None
    setting: str = None
    fold_id: int = None
    prompt_id: int = None
    split: str | list[str] = None

@dataclass
class CustomLoraConfig:
    r: int = field(metadata={"help": "lora rank"})
    lora_alpha: int = field(metadata={"help": "lora alpha"})
    lora_dropout: float = field(metadata={"help": "lora dropout"})
    target_modules: list[str] = field(metadata={"help": "target modules"})
    modules_to_save: list[str] = field(default=None, metadata={"help": "modules to fine-tune and save instead of lora"})
    bias: str = field(default='none')