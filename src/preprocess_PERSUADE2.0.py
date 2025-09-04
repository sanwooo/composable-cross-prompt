import os, re, json
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

def preprocess_df(df: pd.DataFrame):
    # leave only one data for each unique essay id
    df = df.drop_duplicates(['essay_id_comp'])
    
    # drop unnecessary columns
    columns_to_keep = ['essay_id_comp', 'task', 'prompt_name', 'assignment', 'full_text', 'holistic_essay_score', 'essay_word_count']
    df = df.loc[:, columns_to_keep]
    df.rename({
        'essay_id_comp': 'essay_id',
        'full_text': 'essay',
        'holistic_essay_score': 'score',
    }, inplace=True, axis=1)
    df['essay_id'] = np.arange(len(df))
    return df


def read_data_and_template(dataset_path_list: list[str], template_path: str) -> pd.DataFrame:
    assert len(dataset_path_list) == 2

    df_train = pd.read_csv(dataset_path_list[0], low_memory=False)
    df_test = pd.read_csv(dataset_path_list[1], low_memory=False)

    df_train = preprocess_df(df_train)
    df_test = preprocess_df(df_test)
    df_data = pd.concat([df_train, df_test], axis=0)

    df_template = pd.read_excel(template_path, na_filter=False)
    # score column type: float -> int
    df_data['score'] = df_data['score'].astype(int)

    df = df_template.merge(df_data, how='inner', on='prompt_name')
    columns_to_keep = ['essay_id', 'prompt_id', 'prompt', 'essay', 'score', 'min_score', 'max_score', \
                       'user_msg_tmp', 'assistant_msg_tmp']
    df = df.loc[:, columns_to_keep]
    return df

def cleanse(x: str) -> str:
    """
        additional pre=processing for a string.
    """
    return x.strip()

def fill_up_user_message_template(
        user_msg_tmp,
        prompt,
        essay,
        min_score,
        max_score,
):
    user_msg_tmp = re.sub(r"{prompt}", prompt, user_msg_tmp)
    user_msg_tmp = re.sub(r"{essay}", cleanse(essay), user_msg_tmp)
    user_msg_tmp = re.sub(r"{min_score}", str(int(min_score)), user_msg_tmp)
    user_msg_tmp = re.sub(r"{max_score}", str(int(max_score)), user_msg_tmp)
    return user_msg_tmp

def fill_up_assistant_message_template(
        assistant_msg_tmp,
        score,
):
    assistant_msg_tmp = re.sub(r"{score}", str(int(score)), assistant_msg_tmp)
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

def convert_to_conversation_format(df: pd.DataFrame, path: str):
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

def split_dataset_cross_prompt(df: pd.DataFrame, prompt_id: int, dev_ratio) -> tuple[pd.DataFrame, pd.DataFrame]:
    # use prompt [1, 3, 5, 7, 9, 11, 13, 15] only
    other_prompts = [x for x in [1, 3, 5, 7, 9, 11, 13, 15] if x != prompt_id]

    df_train_dev = df.loc[df['prompt_id'].isin(other_prompts)]
    df_train, df_dev = train_test_split(df_train_dev, test_size=dev_ratio, shuffle=True, random_state=42)

    df_test = df.loc[df['prompt_id'] == prompt_id]
    return df_train, df_dev, df_test

def get_split_dataset_cross_prompt(df: pd.DataFrame, dev_ratio: float=0.2) -> None:
    
    for prompt_id in range(1, 16, 2):
        df_train, df_dev, df_test  = split_dataset_cross_prompt(df, prompt_id, dev_ratio=dev_ratio)
        folder_path = os.path.join(f"./assets/PERSUADE2.0/cross-prompt/prompt_{prompt_id}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df_train.to_csv(os.path.join(folder_path, "train.tsv"), sep='\t', index=False)
        df_dev.to_csv(os.path.join(folder_path, "dev.tsv"), sep='\t', index=False)
        df_test.to_csv(os.path.join(folder_path, "test.tsv"), sep='\t', index=False)

        convert_to_conversation_format(df_train, os.path.join(folder_path, "train.jsonl"))
        convert_to_conversation_format(df_dev, os.path.join(folder_path, "dev.jsonl"))
        convert_to_conversation_format(df_test, os.path.join(folder_path, "test.jsonl"))


if __name__ == '__main__':

    df = read_data_and_template(
        dataset_path_list=[
            "./assets/PERSUADE2.0/persuade_corpus_2.0_train.csv",
            "./assets/PERSUADE2.0/persuade_corpus_2.0_test.csv",
        ],
        template_path="./assets/PERSUADE2.0/sft_template_simple.xlsx",
    )
    df = fill_up_chat_columns(df)
    get_split_dataset_cross_prompt(df)