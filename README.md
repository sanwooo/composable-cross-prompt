# Composable Cross-prompt Essay Scoring by Merging Models

This is the code implementation of the EMNLP 2025 paper 
"Composable Cross-prompt Essay Scoring by Merging Models" by Sanwoo Lee, Kun Liang and Yunfang Wu*, Peking University.

## Dependencies

To run the code, please install [uv](https://docs.astral.sh/uv/getting-started/installation/) and install all dependencies specified in `pyproject.toml` using the following command
```bash
uv sync
```

## Step 1. Datasets & Pre-processing

Download ASAP dataset from https://www.kaggle.com/competitions/asap-aes/data, extract files, and place `training_set_rel3.xlsx` under the `assets/ASAP` directory. 
Next, run pre-processing:
```bash
python src/preprocess_ASAP.py
```


Download PERSUADE2.0 dataset from https://github.com/scrosseye/persuade_corpus_2.0, and place `persuade_corpus_2.0_train.csv` and `persuade_corpus_2.0_test.csv` under `assets/PERSUADE2.0` directory.
Next, run pre-processing:
```bash
python src/preprocess_PERSUADE2.0.py
```

## Step 2. SFT
To merge models, we first need to do supervised fine-tuning (SFT) using LoRA on each prompt of ASAP individually, by the following command: 
```bash
export CUDA_VISIBLE_DEVICES=0

experiment_name="{any_name_for_your_experiment}"
dataset_name="ASAP"
setting="cross-prompt-individual"
model_dir="{path_to_directory_which_contains_the_LLM}" 
model_name="Phi-4-mini-instruct" # supports Llama-3.1-8B-Instruct and Phi-4-mini-instruct
config_path="./configs/sft_cross-prompt-individual.yaml"
early_stopping_patience=3

for seed in 42
    do
    for prompt_id in 1 2 3 4 5 6 7 8
        do
        python src/sft.py --experiment_name $experiment_name --seed $seed \
        --dataset_name $dataset_name --setting $setting --prompt_id $prompt_id \
        --model_dir $model_dir --model_name $model_name --config_path $config_path \
        --early_stopping_patience $early_stopping_patience
    done
done
```

This will save the LoRA checkpoint at `./experiments/sft/{$experiment_name}/ckpt/{$model_name}/ASAP/cross-prompt-individual/prompt_{$prompt_id}/seed_{$seed}`.


<details>
    <summary>(Optional) SFT jointly on all source prompts</summary>
    Optionally, if you wish to do SFT jointly on all source prompt datasets for each target prompt, run:
    
    export CUDA_VISIBLE_DEVICES=0
    
    experiment_name="{sft_experiment_name}"
    dataset_name="ASAP"
    setting="cross-prompt"
    model_dir="{path_to_directory_which_contains_the_LLM}" 
    model_name="Phi-4-mini-instruct" # supports Llama-3.1-8B-Instruct and Phi-4-mini-instruct
    config_path="./configs/sft_cross-prompt.yaml"
    early_stopping_patience=10
    
    for seed in 42
        do
        for prompt_id in 1 2 3 4 5 6 7 8
            do
            python src/sft.py --experiment_name $experiment_name --seed $seed \
            --dataset_name $dataset_name --setting $setting --prompt_id $prompt_id \
            --model_dir $model_dir --model_name $model_name --config_path $config_path \
            --early_stopping_patience $early_stopping_patience
        done
    done

This will save the LoRA checkpoint at `./experiments/sft/{$experiment_name}/ckpt/{$model_name}/ASAP/cross-prompt/prompt_{$prompt_id}/seed_{$seed}`.
    
</details>





## Step 3. domain-adaptive model merging

### in-dataset (ASAP -> ASAP) cross-prompt essay scoring

```bash
export CUDA_VISIBLE_DEVICES=0

experiment_name="{merge_experiment_name}"
sft_experiment_name="{experiment_name_used_for_sft (step 2)}"
model_dir="{path_to_directory_which_contains_the_LLM}" 
model_name="Phi-4-mini-instruct" # supports Llama-3.1-8B-Instruct and Phi-4-mini-instruct
ckpt_root="./experiments/sft/"$sft_experiment_name"/ckpt/"$model_name"/ASAP/cross-prompt-individual"
batch_size=4

dataset_name=ASAP
for seed in 42
    do
    for prompt_id in 1 2 3 4 5 6 7 8
        do
        python src/pim_merge.py --experiment_name $experiment_name --seed $seed \
        --model_dir $model_dir --model_name $model_name --ckpt_root $ckpt_root \
        --dataset_name $dataset_name --prompt_id $prompt_id \
        --batch_size $batch_size
    done
done
```
This will save the evaluation results (including QWK) at 
`./experiments/pim_merge/{$experiment_name}/sft_{$sft_experiment_name}_ckpt_{$model_name}_ASAP_cross-prompt-individual/{$dataset_name}/cross-prompt/prompt_{$prompt_id}/seed_{$seed}/greedy.json`

> [!NOTE]
> Please make sure that `sft_experiment_name` matches the one used in step 2 (SFT).

This will save the evaluation result (the merged model is not saved) at directory `experiments/df-merge/$experiment_name/metrics/$model_name_or_path/$acquisition_fn/use_fisher_True/seed$seed.`

### cross-dataset (ASAP -> PERSUADE2.0) cross-prompt essay scoring

The command is same as in-dataset cross-prompt experiment, except that you should make the following modification:
```bash
dataset_name=PERSUADE2.0
for seed in 42
    do
    for prompt_id in 1 3 5 7 9 11 13 15 # (these corresponds to prompt_id 1 2 3 4 5 6 7 8 of PERSUADE2.0 in the paper)
        do
        ...
    ...
```



