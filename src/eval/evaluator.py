import os, re, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score

from utils.configs import DatasetConfig
from data.processor import EvalDataset
from utils.constants import SCORE_RANGE

class Evaluator:
    def __init__(
            self,
            model: AutoModelForCausalLM,
            dataset_config: DatasetConfig,
            tokenizer: AutoTokenizer,
            batch_size: int,
            metric: str='greedy',
    ):
        self.model = model
        self.model.eval()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left' # A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
        self.tokenizer.truncation_side = 'left'
        self.batch_size = batch_size
        self.metric = metric

        if self.metric == 'greedy':
            self.generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id = self.tokenizer.pad_token_id,
            )
        else: # Regression-Aware Inference for LLMs
            self.generation_config = GenerationConfig(
                do_sample=True,
                max_new_tokens=1024,
                temperature=1.0,
                top_k=40,
                num_return_sequences=8,
                pad_token_id = tokenizer.pad_token_id,
            )
        
        self.eval_result = None

    def _row_modes_random_choice(self, arr):
        """
        Compute the mode for each row of a 2D numpy array.
        If there are multiple modes, choose one randomly.
        
        Parameters:
            arr (numpy.ndarray): Input 2D numpy array of integers.
        
        Returns:
            numpy.ndarray: An array of modes for each row.
        """
        batch_size, k = arr.shape
        modes = []
        
        for i in range(batch_size):
            row = arr[i]
            counts = np.bincount(row)
            max_count = counts.max()
            tied_modes = np.flatnonzero(counts == max_count)  # Indices of modes with max frequency
            
            if len(tied_modes) > 1:  # There are ties
                modes.append(np.random.choice(tied_modes))
            else:
                modes.append(tied_modes[0])
        
        return np.array(modes)
    
    def _str_to_y(self, str_y: str):
        """
            convert string-formatted y (score) into integer y
            output: int if convertible else None
        """
        str_y = re.search(r'\d+', str_y)
        if str_y:
            return int(str_y.group(0))
        else:
            return np.nan
        
    def _digitize(
            self,
            y_pred_batch: np.array,
            min_score: int,
            max_score: int,
    ):
        bins = np.linspace(min_score, max_score, num=max_score-min_score+1, endpoint=False)
        bin_indicies = np.digitize(y_pred_batch, bins=bins) - 1
        digitized_y_pred = bin_indicies + min_score
        return digitized_y_pred.tolist()

    @torch.inference_mode()
    def _inference(
            self,
            inputs: dict[str, torch.Tensor],
    ) -> np.array:
        batch_size = inputs['input_ids'].shape[0]
        outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=self.generation_config,
            )
        str_y_list = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        y_list = [self._str_to_y(str_y) for str_y in str_y_list]
        y_array = np.array(y_list).reshape(batch_size, -1) # (batch_size, K)
        if self.metric == 'greedy':
            y_pred_batch = y_array.reshape(-1)
        elif self.metric == 'classification':
            y_pred_batch = self.row_modes_random_choice(y_array)
        elif self.metric == 'regression':
            y_pred_batch = np.nanmean(y_array, axis=-1)
        elif self.metric == 'ordinal_regression':
            y_pred_batch = np.nanmedian(y_array, axis=-1)
        return y_pred_batch.tolist()
    
    def evaluate(
            self,
    ):
        start_time = time.time()
        test_loader = EvalDataset(self.dataset_config).load_data_loader(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
        )

        result = {
            'statistics': {
                'n': None,
                'nan_n': None,
                'inference_time': None,
            },
            'metric': {
                'qwk': None,
            },
            'prediction': {
                'essay_id': [],
                'y_pred': [],
                'y_true': [],
            }
        }

        min_score, max_score = SCORE_RANGE[self.dataset_config.dataset_name][self.dataset_config.prompt_id][0], SCORE_RANGE[self.dataset_config.dataset_name][self.dataset_config.prompt_id][1]
        for batch in test_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            y_pred_batch = self._inference(batch)
            if self.metric in ['regression', 'ordinal_regression']:
                y_pred_batch = self._digitize(y_pred_batch, min_score, max_score)

            result['prediction']['essay_id'] += batch['essay_id'].cpu().tolist()
            result['prediction']['y_true'] += batch['score'].cpu().tolist()
            result['prediction']['y_pred'] += y_pred_batch
          
        
        y_preds, y_trues = np.array(result['prediction']['y_pred']), np.array(result['prediction']['y_true'])
        n = len(y_trues)
        nan_mask = np.isnan(y_preds) | (y_preds < min_score) | (y_preds > max_score)
        nan_n = int(np.sum(nan_mask))
        y_preds = y_preds[~nan_mask]
        y_trues = y_trues[~nan_mask]

        qwk = cohen_kappa_score(
            y1=y_preds,
            y2=y_trues,
            weights='quadratic',
            labels=np.arange(min_score, max_score+1),
        )
        result['metric']['qwk'] = round(qwk, 3)
        result['statistics']['n'] = n
        result['statistics']['nan_n'] = nan_n
        result['statistics']['inference_time'] = time.time() - start_time
        self.eval_result = result
        return result

    def save_eval_result(
            self,
            save_dir: str,
    ):
        assert self.eval_result is not None, "evaluate first"
        save_path = os.path.join(
            save_dir,
            f"{self.metric}.json"    
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(self.eval_result, f, indent=4)