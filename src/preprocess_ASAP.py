import os, re, json
import numpy as np
import pandas as pd
from functools import reduce

def read_data_and_template(dataset_path: str, template_path: str) -> pd.DataFrame:
    df_data = pd.read_excel(dataset_path)
    df_template = pd.read_excel(template_path, na_filter = False)
    df_data.rename({
        'essay_set': 'prompt_id',
        'domain1_score': 'score',
    }, axis=1, inplace=True)
    # missing score value for essay id 10534 , this is inferred from training_set_rel3.tsv
    df_data.loc[df_data['essay_id'] == 10534, 'score'] = 1
    # score column type: float -> int
    df_data['score'] = df_data['score'].astype(int)
    df = df_template.merge(df_data, how='inner', on='prompt_id')
    df = df.loc[:, ['essay_id', 'prompt_id', 'prompt', 'essay', 'score', 'min_score', 'max_score',\
                    'user_msg_tmp', 'assistant_msg_tmp']]
    return df

def cleanse(x: str) -> str:
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # clean essay
    x = x.strip()
    # Remove '@name'
    x = re.sub(r'(@.*?)[\s]', ' ', x)
    # Replace '&amp;' with '&'
    x = re.sub(r'&amp;', '&', x)
    # Remove trailing whitespace
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def fill_up_user_message_template(
        user_msg_tmp: str,
        prompt: str,
        essay: str,
        min_score: int,
        max_score: int,
) -> str:
    user_msg_tmp = re.sub(r'{prompt}', prompt, user_msg_tmp)
    user_msg_tmp = re.sub(r'{essay}', cleanse(essay), user_msg_tmp)
    user_msg_tmp = re.sub(r'{min_score}', str(int(min_score)), user_msg_tmp)
    user_msg_tmp = re.sub(r'{max_score}', str(int(max_score)), user_msg_tmp)
    return user_msg_tmp

def fill_up_assistant_message_template(
        assistant_msg_tmp: str,
        score: int,
) -> str:
    assistant_msg_tmp = re.sub(r'{score}', str(int(score)), assistant_msg_tmp)
    return assistant_msg_tmp

def fill_up_chat_columns(df: pd.DataFrame) -> pd.DataFrame:

    df['essay'] = df['essay'].map(lambda x: str(x).replace('\\','\\\\')) # to avoid bad escape \e at position xxxx error
    df['user'] = df.apply(lambda x: fill_up_user_message_template(
        x['user_msg_tmp'],
        x['prompt'],
        x['essay'],
        x['min_score'],
        x['max_score'],
    ), axis=1)
    df['assistant'] = df.apply(lambda x: fill_up_assistant_message_template(
        x['assistant_msg_tmp'],
        x['score'],
    ), axis=1)
    return df

def convert_to_conversation_format(df: pd.DataFrame, path: str) -> None:
    """
        convert a pd.DataFrame with columns: ['user', 'assistant'] into conversational format (jsonl),
        to fit SFTTrainer. 
    """
    template_list = []
    for i, instance in df.iterrows():
        template = {
            "messages": [
                {
                    "role": "user",
                    "content": instance["user"],
                },
                {
                    "role": "assistant",
                    "content": instance["assistant"],
                }
            ]
        }
        template = json.dumps(template)
        template_list.append(template)

    with open(path, "w") as f:
        for temp in template_list:
            json.loads(temp)
            f.write(temp + "\n")
    
    return


#### cross-prompt
def split_dataset_cross_prompt(df: pd.DataFrame, prompt_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        split dataset (pd.DataFrame) for multi-source (7 for ASAP) cross-prompt, using Ridley (2021) train-dev-test split.
    """
    train_ids = pd.read_csv(f'./assets/ASAP/ridley_split/{prompt_id}/train.tsv', sep='\t')['essay_id'].astype(int).tolist()
    dev_ids = pd.read_csv(f'./assets/ASAP/ridley_split/{prompt_id}/dev.tsv', sep='\t')['essay_id'].astype(int).tolist()
    test_ids = pd.read_csv(f'./assets/ASAP/ridley_split/{prompt_id}/test.tsv', sep='\t')['essay_id'].astype(int).tolist()

    df_train = df.loc[df['essay_id'].isin(train_ids)]
    df_dev = df.loc[df['essay_id'].isin(dev_ids)]
    df_test = df.loc[df['essay_id'].isin(test_ids)]

    return df_train, df_dev, df_test

def get_split_dataset_cross_prompt(df: pd.DataFrame):

    for prompt_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        df_train, df_dev, df_test = split_dataset_cross_prompt(df, prompt_id=prompt_id)
        folder_path = f'./assets/ASAP/cross-prompt/prompt_{prompt_id}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df_train.to_csv(os.path.join(folder_path, "train.tsv"), sep='\t', index=False)
        df_dev.to_csv(os.path.join(folder_path, "dev.tsv"), sep='\t', index=False)
        df_test.to_csv(os.path.join(folder_path, "test.tsv"), sep='\t', index=False)

        convert_to_conversation_format(df_train, os.path.join(folder_path, "train.jsonl"))
        convert_to_conversation_format(df_dev, os.path.join(folder_path, "dev.jsonl"))
        convert_to_conversation_format(df_test, os.path.join(folder_path, "test.jsonl"))
        
    return

#### cross-prompt-individual
def split_dataset_cross_prompt_individual(df: pd.DataFrame, prompt_id: int):

    candidate_train_dfs = [pd.read_csv(f'./assets/ASAP/ridley_split/{x}/train.tsv', sep='\t')\
                           for x in [1,2,3,4,5,6,7,8] if x != prompt_id]
    candidate_dev_dfs = [pd.read_csv(f'./assets/ASAP/ridley_split/{x}/dev.tsv', sep='\t')\
                           for x in [1,2,3,4,5,6,7,8] if x != prompt_id]
    candidate_train_ids = [set(x.loc[x['essay_set']==prompt_id, 'essay_id'].astype(int).tolist()) for x in candidate_train_dfs]
    candidate_dev_ids = [set(x.loc[x['essay_set']==prompt_id, 'essay_id'].astype(int).tolist()) for x in candidate_dev_dfs]

    intersection_train = reduce(set.intersection, candidate_train_ids)
    intersection_dev = reduce(set.intersection, candidate_dev_ids)

    # a total of 7 labeled samples are lost during the above intersection.
    # lost essay_id: prompt_1 {26}, prompt_3 {6260}, prompt_id 4 {9331}, prompt_id 5 {13631, 12735}, prompt_id 6 {15074, 16590}
    # we don't add them back to the train/dev set of cross_prompt_individual split, to ensure that 
    # our model merging approach does not take advantage of more labeled samples over cross_prompt (joint-training) setting. 

    df_train = df.loc[df['essay_id'].isin(intersection_train)]
    df_dev = df.loc[df['essay_id'].isin(intersection_dev)]

    return df_train, df_dev

def get_split_dataset_cross_prompt_individual(df):

    for prompt_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        df_train, df_dev = split_dataset_cross_prompt_individual(df, prompt_id=prompt_id)
        folder_path = f'./assets/ASAP/cross-prompt-individual/prompt_{prompt_id}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df_train.to_csv(os.path.join(folder_path, "train.tsv"), sep='\t', index=False)
        df_dev.to_csv(os.path.join(folder_path, "dev.tsv"), sep='\t', index=False)

        convert_to_conversation_format(df_train, os.path.join(folder_path, "train.jsonl"))
        convert_to_conversation_format(df_dev, os.path.join(folder_path, "dev.jsonl"))
    return

if __name__ == '__main__':
    # prompt-specific
    df = read_data_and_template(dataset_path='./assets/ASAP/training_set_rel3.xlsx', \
                           template_path='./assets/ASAP/sft_template_simple.xlsx')
    df = fill_up_chat_columns(df)
    get_split_dataset_cross_prompt(df)
    get_split_dataset_cross_prompt_individual(df)
