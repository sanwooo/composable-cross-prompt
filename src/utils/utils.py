import os
import shutil
import re
from datetime import datetime
from dataclasses import fields
from shutil import copytree
from utils.logger import logger, log_info_file
from utils.constants import PROMPT_ID_LIST
from utils.configs import DatasetConfig

def create_experiment_directory(
        project_root: str,
        file_name: str,
        experiment_name: str,
) -> str:
    # create unique experiment directory
    experiment_dir = os.path.join(project_root, "experiments")
    # filename + run_name (since there can be many versions for a single python file)
    if experiment_name:
        experiment_dir = os.path.join(experiment_dir, file_name, experiment_name)
    else:
        now = datetime.now()
        timestamp = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        experiment_dir = os.path.join(experiment_dir, file_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # save current source code and configuration in the experiment directory. 
    for backup_dirname in ["src", "configs"]:
        copytree(
            os.path.join(project_root, backup_dirname),
            os.path.join(experiment_dir, backup_dirname),
            dirs_exist_ok=True,
        )
    # synchronize logger output to run.log file
    log_info_file(os.path.join(experiment_dir, 'run.log'))
    return experiment_dir

def update_dataclass_with_args(dataclass, args):
    for attribute in fields(dataclass):
        if hasattr(args, attribute.name):
            setattr(dataclass, attribute.name, getattr(args, attribute.name))
    return

def delete_subdirs_with_regex(dir_path, regex):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        # Check if the item is a directory and matches the regex pattern
        if os.path.isdir(item_path) and re.match(regex, item):
            print(f"Deleting directory: {item_path}")
            shutil.rmtree(item_path)  # Recursively delete the directory

def infer_merge_prompt_ids(
        src_ckpt_root: str,
        tgt_dataset_config: DatasetConfig,
) -> list[int]:
    """
        if src_dataset_name == tgt_dataset_name, merge_prompt_ids becomes all prompt_ids except for the target prompt id
        if src_dataset_name != tgt_dataset_name, merge_prompt_ids becomes all prompt_ids of the source dataset
    """
    dataset_name_list = list(PROMPT_ID_LIST.keys())
    src_dataset_name = re.search("|".join(dataset_name_list), src_ckpt_root).group()
    # in-dataset cross-prompt
    tgt_dataset_name = tgt_dataset_config.dataset_name
    if src_dataset_name == tgt_dataset_name and tgt_dataset_config.setting == 'cross-prompt' and tgt_dataset_config.split == 'test':
        merge_prompt_ids = [x for x in PROMPT_ID_LIST[src_dataset_name] if x != tgt_dataset_config.prompt_id]
    # in-dataset prompt-specific
    elif src_dataset_name == tgt_dataset_name and tgt_dataset_config.setting == 'cross-prompt-individual' and tgt_dataset_config.split == 'dev':
        merge_prompt_ids = PROMPT_ID_LIST[src_dataset_name]
    # cross-datasets cross-prompt
    elif src_dataset_name != tgt_dataset_name:
        merge_prompt_ids = PROMPT_ID_LIST[src_dataset_name]
    else:
        raise NotImplementedError
    return src_dataset_name, tgt_dataset_name, merge_prompt_ids