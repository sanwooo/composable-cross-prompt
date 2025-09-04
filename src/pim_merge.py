import os
import argparse
import re
from copy import deepcopy
import time

import numpy as np
from scipy.stats import beta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraModel

from bayes_opt import BayesianOptimization, acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from utils.constants import BETA_STATISTICS, SCORE_RANGE
from utils.logger import logger
from utils.configs import DatasetConfig
from utils.utils import create_experiment_directory, update_dataclass_with_args, infer_merge_prompt_ids
from data.processor import DatasetBase, EvalDataset
from model.utils import load_adapters
from model.mergers import LearnableLoraMerge
from eval.evaluator import Evaluator


def setup_experiment():
    cmd_parser = argparse.ArgumentParser()
    # general config
    cmd_parser.add_argument("--experiment_name", type=str, default=None)
    cmd_parser.add_argument("--seed", type=int, default=42)

    # checkpoint config
    cmd_parser.add_argument("--model_dir", type=str)
    cmd_parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Phi-4-mini-instruct"])
    cmd_parser.add_argument("--ckpt_root", type=str)

    # eval dataset config
    cmd_parser.add_argument("--dataset_name", type=str, choices=["ASAP", "PERSUADE2.0"])
    cmd_parser.add_argument("--prompt_id", type=int)
    cmd_parser.add_argument("--setting", type=str, default="cross-prompt")
    cmd_parser.add_argument("--split", type=str, default='test')

    # inference config
    cmd_parser.add_argument("--metric", type=str, default="greedy")
    cmd_parser.add_argument("--batch_size", type=int, default=4)
    
    # method-specific config
    cmd_parser.add_argument("--acquisition_fn", type=str, default='ei', choices=['ei', 'ucb'])
    cmd_parser.add_argument("--init_points", type=int, default=10)
    cmd_parser.add_argument("--n_iter", type=int, default=30)
    cmd_parser.add_argument("--n_sample", type=int, default=64, help="number of samples to use from the test set.")

    cmd_args = cmd_parser.parse_args()

    # create experiment_directory
    experiment_dir = create_experiment_directory(
        project_root=".",
        file_name=os.path.splitext(os.path.basename(__file__))[0],
        experiment_name=cmd_args.experiment_name,
    )
    # update configs
    dataset_config = DatasetConfig()
    update_dataclass_with_args(dataset_config, cmd_args)

    # create output_dir
    dataset_identifier = DatasetBase(dataset_config=dataset_config).infer_unique_identifier()
     
    ckpt_root_identifier = "_".join(cmd_args.ckpt_root.split("/")[-6:])
    cmd_args.output_dir = os.path.join(
        experiment_dir,
        ckpt_root_identifier,
        dataset_identifier,
        f"seed_{cmd_args.seed}",
    )
    os.makedirs(cmd_args.output_dir, exist_ok=True)

    logger.info(f"cmd_args: {vars(cmd_args)}\n")
    logger.info(f"dataset_config: {vars(dataset_config)}\n")
    return cmd_args, dataset_config

class BayesianMergePIM:
    """
        prior-enhanced modified information maximization objective
    """

    def __init__(
            self,
            model: LoraModel,
            tokenizer: AutoTokenizer,
            src_dataset_name: str,
            tgt_dataset_config: DatasetConfig,
            n_sample: int,
            batch_size: int,
    ):
        self.src_adapter_names = list(model.peft_config.keys())
        self.src_dataset_name = src_dataset_name
        
        self.src_pid_list = [int(re.search(r'\d+', x).group(0)) for x in self.src_adapter_names]

        self.score_list = list(range(
            SCORE_RANGE[tgt_dataset_config.dataset_name][tgt_dataset_config.prompt_id][0],
            SCORE_RANGE[tgt_dataset_config.dataset_name][tgt_dataset_config.prompt_id][1]+1,
        ))
        self.score_tok_ids = tokenizer.convert_tokens_to_ids([str(x) for x in self.score_list])        
        self.score_tok_pos = -1 # dataset assumes that the last token is responsible for predicting score
        
        self.src_marginal = self._merge_beta_mixture(src_dataset_name, self.src_pid_list)
    
        self.merge_wrapper = LearnableLoraMerge(
            lora_model=model,
            level='modelwise',
            last_k_layers=None,
        )

        self.tgt_dataset_config = tgt_dataset_config
        self.tgt_data_loader = EvalDataset(tgt_dataset_config).load_data_loader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle=False,
            n_sample=n_sample,
        )

    def _merge_beta_mixture(
            self,
            src_dataset_name: str,
            src_pid_list: list[int],
    ):
        """
            combine a beta mixture (1/M)Sum_{i=1}^{M} Beta(alpha_i, beta_i) into a single Beta
            such that its mean and variance match the mean and variance of the mixture.
        """
        mean_list, var_list = [], []
        for src_pid in src_pid_list:
            alpha_, beta_ = BETA_STATISTICS[src_dataset_name][src_pid]
            dist = torch.distributions.Beta(alpha_, beta_)
            mean, var = dist.mean, dist.variance
            mean_list.append(mean)
            var_list.append(var)
        mean_tensor = torch.Tensor(mean_list)
        var_tensor = torch.Tensor(var_list)
        mixture_mean = torch.mean(mean_tensor)
        mixture_var = torch.mean(var_tensor) + torch.mean(mean_tensor**2) - mixture_mean**2
        new_alpha = mixture_mean * ((mixture_mean * (1-mixture_mean) / mixture_var) -1)
        new_beta = (1-mixture_mean) * ((mixture_mean * (1-mixture_mean) / mixture_var) -1)
        beta_dist = beta(new_alpha.item(), new_beta.item()) # torch does not support cdf for beta, we use scipy instead here
        bin_edges = np.linspace(0, 1, num=len(self.score_list)+1)
        bin_probs = beta_dist.cdf(bin_edges[1:]) - beta_dist.cdf(bin_edges[:-1])
        src_marginal = torch.Tensor(bin_probs).cuda()
        return src_marginal
    
    @torch.no_grad()
    def _collect_predictions(
            self,
            data_loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        norm_score_probs_list = []
        for batch in data_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = self.merge_wrapper.lora_model(
                **batch,
            )
            probs = F.softmax(output.logits, dim=-1) 
            score_probs = probs[:, self.score_tok_pos, self.score_tok_ids].float() # (batch_size, score_n)
            # normalize
            norm_score_probs = score_probs / score_probs.sum(dim=-1).view(-1, 1) # (batch_size, score_n)
            norm_score_probs_list.append(norm_score_probs)
        
        return torch.concat(norm_score_probs_list, dim=0)
    
    def _entropy(
            self,
            x: torch.Tensor, # probs (batch_size, score_n)
            eps: float=1e-8,
    ):
        entropy_batch = -torch.sum(x * torch.log(x+eps), dim=-1) # (batch_size, )
        entropy = entropy_batch.mean()
        return entropy
    
    def _source_kl_div(
            self,
            x: torch.Tensor,
            eps: float=1e-8,
    ):
        x_mean = torch.mean(x, dim=0) # (score_n, )
        q = x_mean + eps # (score_n, )
        p = self.src_marginal + eps # (score_n, )
        kl = (p * torch.log(p / q)).sum()
        return kl
    
    @torch.no_grad()
    def _black_box_function(
            self,
            coef_dict: dict[str, float],
    ):
        coefs = torch.tensor(list(coef_dict.values()), dtype=torch.float32).cuda()
        self.merge_wrapper.merge(coefs=coefs)
        y_prob = self._collect_predictions(self.tgt_data_loader)

        kl_divergence = self._source_kl_div(y_prob)
        entropy = self._entropy(y_prob)

        obejctive = (-1) * entropy + (-1) * kl_divergence
        logger.info(f"entropy: {entropy:.3f}  kl_divergence: {kl_divergence:.3f}  objective: {obejctive:.3f}")
        return obejctive.item()

    @torch.no_grad()
    def merge(
            self,
            acquisition_fn: acquisition.ExpectedImprovement | acquisition.UpperConfidenceBound,
            seed: int,
            init_points: int,
            n_iter: int,
            log_path: str,
    ):
        pbounds = {adapter_name: (0, 1) for adapter_name in self.src_adapter_names}
        optimizer = BayesianOptimization(
            f=lambda **coef_dict: self._black_box_function(
                coef_dict,
            ),
            pbounds=pbounds,
            acquisition_function=acquisition_fn,
            random_state=seed,
            verbose=2,
        )

        bayes_opt_logger = JSONLogger(path=log_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, bayes_opt_logger)
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        optimal_coefs =optimizer.max['params']
        self.optimal_coefs = list(optimal_coefs.values())
        optimal_coefs = torch.tensor(list(optimal_coefs.values()), dtype=torch.float32).cuda()
        logger.info(f"optimal_coef: {optimal_coefs}")
        self.merge_wrapper.merge(optimal_coefs)
        return self.merge_wrapper.lora_model


if __name__ == '__main__':

    args, tgt_dataset_config = setup_experiment()

    src_dataset_name, tgt_dataset_name, merge_prompt_ids = infer_merge_prompt_ids(args.ckpt_root, tgt_dataset_config)
    model, tokenizer, response_template = load_adapters(
        model_dir=args.model_dir,
        model_name=args.model_name,
        ckpt_root=args.ckpt_root,
        merge_prompt_ids=merge_prompt_ids,
        seed=args.seed,
    )
    model.eval()
    
    merge_wrapper = BayesianMergePIM(
        model=model,
        tokenizer=tokenizer,
        src_dataset_name=src_dataset_name,
        tgt_dataset_config=tgt_dataset_config,
        n_sample=args.n_sample,
        batch_size=args.batch_size,
    )

    if args.acquisition_fn == 'ei':
        acquisition_fn = acquisition.ExpectedImprovement(xi=0.01)
    elif args.acquisition_fn == 'ucb':
        acquisition_fn = acquisition.UpperConfidenceBound(kappa=2.576)

    log_path = os.path.join(
            args.output_dir,
            args.acquisition_fn,
            "log.log",
        )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    start_time = time.time()
    merged_model = merge_wrapper.merge(
        acquisition_fn=acquisition_fn,
        seed=args.seed,
        init_points=args.init_points,
        n_iter=args.n_iter,
        log_path=log_path,
    )
    elapsed_time = time.time() - start_time 

    evaluator = Evaluator(
        model=merged_model,
        dataset_config=tgt_dataset_config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        metric=args.metric,
    )
    evaluator.evaluate()
    evaluator.eval_result['statistics']['adaptation_time'] = elapsed_time
    evaluator.eval_result['statistics']['optimal_coef'] = merge_wrapper.optimal_coefs
    evaluator.save_eval_result(
        args.output_dir,
    )
