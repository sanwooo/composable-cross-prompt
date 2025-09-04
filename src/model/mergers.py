from dataclasses import asdict, replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraModel
from peft.tuners.lora import LoraLayer, LoraModel, LoraConfig
from peft.utils.other import _freeze_adapter, _get_submodules
from utils.logger import logger

class LoraMergeBase(nn.Module):

    def __init__(
            self,
            lora_model: LoraModel,
            tgt_adapter_name: str='merge',
    ):
        super(LoraMergeBase, self).__init__()
        self.lora_model = lora_model
        self.src_adapter_names = list(lora_model.peft_config.keys())
        self.tgt_adapter_name = tgt_adapter_name

        # inject and set target adapter, assume peft configs are equal across all adapters 
        tgt_peft_config = self.lora_model.peft_config[self.src_adapter_names[0]]
        tgt_peft_config.r = tgt_peft_config.r * len(self.src_adapter_names) # weighted sum of BA_i of rank r has at most r x T rank
        tgt_peft_config.lora_alpha = tgt_peft_config.lora_alpha * len(self.src_adapter_names) # adjust alpha so that alpha / r is kept to the same level as original peft config
        self.lora_model.peft_config[tgt_adapter_name] = tgt_peft_config
        self.lora_model.inject_adapter(self.lora_model.model, adapter_name=self.tgt_adapter_name)
        self.lora_model.set_adapter(self.tgt_adapter_name)

        # freeze adapters
        for adapter_name in self.src_adapter_names + [self.tgt_adapter_name]:
            _freeze_adapter(self.lora_model, adapter_name)

        # collect modules to merge
        self.target_module_dict = self._collect_target_modules(self.lora_model)

    def _collect_target_modules(
            self,
            lora_model: LoraModel,
    ) -> dict[str, LoraLayer]:
        """
            collect LoraLayers from the LoraModel
        """
        key_list = [key for key, _ in lora_model.model.named_modules() if lora_model.prefix not in key]
        target_dict = {}
        for key in key_list:
            _, target, _ = _get_submodules(lora_model.model, key)
            if isinstance(target, LoraLayer):
                target_dict[key] = target
        
        return target_dict

    def _get_loraA_loraB(
            self,
            lora_layer: LoraLayer,
            adapter_name: str,
    )-> tuple[torch.Tensor, torch.Tensor]:
        if self.tgt_adapter_name in lora_layer.lora_A:
            lora_A = lora_layer.lora_A[adapter_name].weight
            lora_B = lora_layer.lora_B[adapter_name].weight
        elif self.tgt_adapter_name in lora_layer.lora_embedding_A:
            lora_A = lora_layer.lora_embedding_A[adapter_name]
            lora_B = lora_layer.lora_embedding_B[adapter_name]
        else:
            raise NotImplementedError
        return lora_A, lora_B

    def merge(
            self,
    ) -> None:
        pass

class LearnableLoraMerge(LoraMergeBase):

    def __init__(
            self,
            lora_model: LoraModel,
            level: str='modelwise',
            last_k_layers: int=None,
            prior: float=0.4,
            tgt_adapter_name: str='merge',
            
    ):
        super(LearnableLoraMerge, self).__init__(lora_model=lora_model, tgt_adapter_name=tgt_adapter_name)
        self.level = level

        # initialize coefficient parameters
        N, L = len(self.src_adapter_names), len(self.target_module_dict)
        if self.level == 'modelwise':
            self.coefs = nn.Parameter(torch.full((N,), fill_value=prior).cuda())
        elif self.level == 'layerwise':
            self.coefs = nn.Parameter(torch.full((L, N), fill_value=prior).cuda())
            self.first_learnable_layer_index = L - last_k_layers if last_k_layers else 0        

    def _get_valid_coefs(self, coefs: nn.Parameter):
        return torch.clamp(coefs, min=0.0, max=1.0)
    
    def collect_learnable_parameters(self):
        return [self.coefs]
    
    def save(self, save_path: str):
        torch.save({'coefs': self.coefs.data}, save_path)
    
    def load(self, ckpt_path: str):
        loaded_parameters = torch.load(ckpt_path, weights_only=True)
        logger.info(f"loaded_parameters:\n{loaded_parameters}")
        self.load_state_dict(loaded_parameters, strict=False)

    def _lora_layer_weighted_sum(
            self,
            lora_layer: LoraLayer,
            coefs: nn.Parameter,
    ):
        # coefs (n_adapters)
        lora_AB_deltas = [self._get_loraA_loraB(lora_layer, x) for x in self.src_adapter_names]
        lora_A_concat = torch.concat([x[0] for x in lora_AB_deltas], dim=0)  # (T * rank, hidden_size_1)
        lora_B_concat = torch.concat([x[1] for x in lora_AB_deltas], dim=1) # list[(hidden_size_2, T * rank)]

        T, Tr = len(self.src_adapter_names), lora_A_concat.shape[0]
        broadcast_coefs = coefs.repeat_interleave(Tr // T) # (T * r)

        # accept negative coefs
        signs = torch.sign(broadcast_coefs)
        sqrt_pos_coefs = torch.sqrt(signs * broadcast_coefs)
        reduced_lora_B = lora_B_concat @ torch.diag(sqrt_pos_coefs)
        reduced_lora_A = torch.diag(signs * sqrt_pos_coefs) @ lora_A_concat

        return reduced_lora_A, reduced_lora_B
    
    def merge(
            self,
            coefs: nn.Parameter,
    ) -> None:
        coefs = self._get_valid_coefs(coefs)
        for layer_idx, (target_key, target) in enumerate(self.target_module_dict.items()):
            target_lora_A, target_lora_B = self._get_loraA_loraB(target, adapter_name=self.tgt_adapter_name)
            target_lora_A.data = target_lora_A.data * 0.
            target_lora_B.data = target_lora_B.data * 0.

            if self.level == "modelwise":
                layer_coefs = coefs
            elif self.level == 'layerwise':
                if layer_idx >= self.first_learnable_layer_index:
                    layer_coefs = coefs[layer_idx, :]
                else:
                    layer_coefs = coefs[layer_idx, :].detach()

            merged_lora_A, merged_lora_B = self._lora_layer_weighted_sum(target, layer_coefs)
            # copy_ retains target tensor (lora_deltas[0])'s requires_grad attribute value and computational graph
            target_lora_A.detach_().copy_(merged_lora_A) 
            target_lora_B.detach_().copy_(merged_lora_B)
        
        return