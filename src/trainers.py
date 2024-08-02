import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch_discounted_cumsum import discounted_cumsum_right, discounted_cumsum_left

import tensor_parallel as tp
import contextlib
# import deepspeed

# from preference_datasets import DataLoader, get_batch_iterator
from dataloader import DataLoader, SFTDataLoader, DPODataLoader, IFTDataLoader, ORPODataLoader

from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
    delete_dict,
    disable_dropout,
)
import numpy as np
import wandb
import tqdm
import matplotlib.pyplot as plt
import math

import gc
import random
import os
import argparse
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

torch.backends.cuda.matmul.allow_tf32 = True

def linear_with_warmup(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    """
    Copied from transformers.optimization._get_linear_schedule_with_warmup_lr_lambda
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )

def cosine_with_warmup(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5
):
    """
    Copied from transformers.optimization._get_cosine_schedule_with_warmup_lr_lambda
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

def linear_warmup(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return 1.0

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    gamma: float = 1.0,
                    label_smoothing: float = 0.0,
                    loss_name: str = 'dpo',
                    reference_free: bool = False,
                    rejected_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    if rejected_free:
        policy_rejected_logps = torch.zeros_like(policy_rejected_logps).to(policy_chosen_logps.device)
        reference_rejected_logps = torch.zeros_like(reference_rejected_logps).to(policy_chosen_logps.device)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_name == 'dpo':
         # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_name == 'ipo':
        # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        losses = (logits - 1/(2 * beta)) ** 2
       
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, 
                     labels: torch.LongTensor, 
                     average_log_prob: bool = False, 
                     per_token_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if per_token_prob:
        return per_token_logps * loss_mask
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

class BasicTrainer(object):
    def __init__(self, 
                 policy: nn.Module, 
                 config: DictConfig, 
                 seed: int, 
                 run_dir: str, 
                 policy_weak: Optional[nn.Module] = None,
                 reference_model: Optional[nn.Module] = None, 
                 truncation_side="right",
                 padding_side="right",
                 rank: int = 0, 
                 world_size: int = 1
        ):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.debug = config.debug
        assert self.config.batch_size % self.config.gradient_accumulation_steps == 0, 'batch_size must be divisible by gradient_accumulation_steps'
        
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            truncation_side=truncation_side,
            padding_side=padding_side,
            cache_dir=get_local_dir(config.local_dirs))
        
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        else:
            rank0_print(f'chat_template: {self.tokenizer.chat_template}')
            
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            loss_name=config.loss.name,
            seed=seed,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )

        self.policy = policy
        self.policy_weak = policy_weak
        self.reference_model = reference_model
        if self.config.loss.name in {'ift'}:
            self.embed_tokens = nn.Embedding(num_embeddings=self.policy.model.embed_tokens.num_embeddings, 
                                            embedding_dim=self.policy.model.embed_tokens.embedding_dim,
                                            padding_idx=self.policy.model.embed_tokens.padding_idx,
                                            device=torch.device(self.rank),
                                            dtype=self.policy.model.embed_tokens.weight.dtype)
            self.embed_tokens.load_state_dict(self.policy.model.embed_tokens.state_dict())
            self.embed_tokens.requires_grad_(False)
        
        self.train_loader = globals()[f'{config.loss.name.upper()}DataLoader'](
            split='train', 
            batch_size=config.batch_size, 
            n_epochs=config.n_epochs, 
            n_examples=config.n_examples, 
            **data_iterator_kwargs
        )
        rank0_print(f'Loaded train data iterator')
        
        self.eval_loader = globals()[f'{config.loss.name.upper()}DataLoader'](
            split='test', 
            batch_size=config.eval_batch_size,
            n_examples=config.n_eval_examples, 
            **data_iterator_kwargs
        )
        self.eval_batches = list(self.eval_loader)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        
        self.train_iterations = len(self.train_loader)
        self.warmup_steps = math.ceil(self.train_iterations * config.warmup_ratio) if config.warmup_ratio is not None else config.warmup_steps
        rank0_print(f'Using {self.warmup_steps} warmup steps')
            
        self.gamma = config.loss.gamma
        self.min_lambda = config.loss.min_lambda
        self.max_lambda = config.loss.max_lambda
        self.lambda_schedule = config.loss.lambda_schedule
        self.lambda_disturb = config.loss.lambda_disturb
        self.disturb_std = config.loss.disturb_std
        
        if self.lambda_disturb == "normal":
            self.noise = []
            for _ in range(self.train_iterations):
                self.noise.append(torch.randn(1).item())
    
    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    
    def get_sample_function(self, logits: torch.FloatTensor, sample: str = 'greedy', k: int = 2, temperature: float = 0.7) -> torch.LongTensor:
        """Get the labels for the given logits using the given sampling strategy."""
        with torch.no_grad():
            if sample == 'greedy':
                token_sample = torch.argmax(logits, dim=-1)
            elif sample == 'topk':
                token_sample = torch.topk(logits, k=k, dim=-1).indices[..., -1]
            elif sample == 'nucleus':
                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.view(-1, vocab_size)
                token_sample = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
                token_sample = token_sample.view(batch_size, seq_len)
            else:
                raise ValueError(f'unknown sample {sample}')
            
        return token_sample
    
    def get_cumsum_weight(
        self, 
        logps, 
        loss_mask, 
        gamma=1,
        propagation_type='loss', 
        propagation_norm='L1', 
        propagation_side='right') -> torch.FloatTensor:
        
        if gamma == 0:
            return torch.ones_like(logps)
        
        if propagation_type == 'mask':
            cumsum_item = loss_mask
        elif propagation_type == 'loss':
            cumsum_item = -logps / loss_mask.sum(-1).unsqueeze(-1)
        elif propagation_type == 'logps':
            cumsum_item = -logps
        else:
            raise ValueError(f'unknown propagation_type {propagation_type}')
        
        cumsum_item[loss_mask == 0] = 0
        
        if propagation_side == 'right':
            if gamma == 1:
                cumsum_weight = torch.cumsum(cumsum_item, dim=-1)
            else:
                cumsum_weight = discounted_cumsum_left(cumsum_item, gamma=gamma)
                
            cumsum_weight[loss_mask == 0] = 1e6
            cumsum_weight += cumsum_weight[cumsum_weight.nonzero(as_tuple=True)].min()
            
            if propagation_norm == 'L1': # sharp level 1
                cumsum_weight = 1 / (cumsum_weight)
            elif propagation_norm == 'L2': # sharp level 2
                cumsum_weight = 1 / (cumsum_weight) ** 2
            elif propagation_norm == 'softmax': # sharp level 3
                cumsum_weight = torch.softmax(1/cumsum_weight, dim=-1)
            elif propagation_norm == 'log': # sharp level 0
                cumsum_weight = 1 / torch.log(cumsum_weight + 1)
            else:
                raise ValueError(f'unknown propagation_norm {propagation_norm}')
            
        elif propagation_side == 'left':
            if gamma == 1:
                cumsum_weight = torch.flip(torch.cumsum(torch.flip(cumsum_item, [-1]), dim=-1), [-1])
            else:
                cumsum_weight = discounted_cumsum_right(cumsum_item, gamma=gamma)
            cumsum_weight[loss_mask == 0] = 0
            
            if propagation_norm == 'L1': # sharp level 2
                cumsum_weight = cumsum_weight
            elif propagation_norm == 'L2': # sharp level 3
                cumsum_weight = cumsum_weight ** 2
            elif propagation_norm == 'softmax': # sharp level 1
                cumsum_weight = torch.softmax(cumsum_weight, dim=-1)
            elif propagation_norm == 'log': # sharp level 0
                cumsum_weight = torch.log(cumsum_weight + 1)
            else:
                raise ValueError(f'unknown propagation_norm {propagation_norm}')
            
        else:
            raise ValueError(f'unknown propagation_side {propagation_side}')
        
        cumsum_weight[loss_mask == 0] = 0
        cumsum_weight *= (loss_mask.sum(-1, keepdim=True) / cumsum_weight.sum(-1, keepdim=True))
        
        return cumsum_weight
    
    def update_lambda(
        self, 
        step_idx: int
    ) -> float:
        """Get the gamma value for the given step index."""
        if self.lambda_schedule:
            schedule = linear_warmup(
                step_idx, 
                num_warmup_steps=self.config.warmup_steps, 
                num_training_steps=self.train_iterations)
            self._lambda = self.min_lambda + (self.max_lambda - self.min_lambda) * schedule
        else:
            self._lambda = self.max_lambda
        
        if self.lambda_disturb:
            self._lambda += self.noise[step_idx] * self.disturb_std * self._lambda
            
        self._lambda = torch.clamp(
            input=torch.tensor(self._lambda), 
            min=0.0, 
            max=1.0).item()
        
        return self._lambda
    
    def debug_inputs(self, inputs, device=0):
        input_ids = inputs['target_input_ids']
        attention_mask = inputs['target_attention_mask']
        labels = inputs['target_labels']
        loss_mask = labels[:, 1:] != -100
        
        if device == "all":
            for input_id, label, mask in zip(input_ids, labels, attention_mask):
                print("#"*10+" input_ids "+"#"*10)
                print(f"{self.tokenizer.decode(input_id)}\n")
                print("#"*10+" labels "+"#"*10)
                print(f"{self.tokenizer.decode(torch.where(label==-100, 0, label))}\n")
                print("#"*10+" attention_mask "+"#"*10)
                print(f"{self.tokenizer.decode(torch.where(mask==0, 0, input_id))}\n")
            
                print(len(input_id))
                print(len(label))
                print(len(attention_mask))

                with open(f"{self.args.output_dir}/{torch.distributed.get_rank()}.json", "w", encoding="utf-8") as file:
                    data_dict = {
                        "input_ids": self.tokenizer.decode(input_id),
                        "labels": self.tokenizer.decode(torch.where(label==-100, 0, label)),
                        "attention_mask": self.tokenizer.decode(torch.where(mask==0, 0, input_id))
                    }
                    json.dump(data_dict, file, indent=4, ensure_ascii=False)


        elif torch.distributed.get_rank() == device:
            for input_id, label, mask in zip(input_ids, labels, attention_mask):
                print("#"*10+" input_ids "+"#"*10)
                print(f"{self.tokenizer.decode(input_id)}\n")
                print("#"*10+" labels "+"#"*10)
                print(f"{self.tokenizer.decode(torch.where(label==-100, 0, label))}\n")
                print("#"*10+" attention_mask "+"#"*10)
                print(f"{self.tokenizer.decode(torch.where(mask==0, 0, input_id))}\n")
            
                print(len(input_id))
                print(len(label))
                print(len(attention_mask))
            
        exit()
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo':
                loss_kwargs = {'loss_name': loss_config.name, 'beta': loss_config.beta, 
                               'reference_free': loss_config.reference_free, 'rejected_free': loss_config.rejected_free, 'label_smoothing': loss_config.label_smoothing}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'loss_name': loss_config.name, 'beta': loss_config.beta}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            
            policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

            all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
            losses = losses.mean()

        elif loss_config.name == 'sft':
            # with KL divergence
            input_ids = batch['target_input_ids']
            attention_mask = batch['target_attention_mask']
            labels = batch['target_labels']
            
            logits = self.policy(input_ids, attention_mask=attention_mask).logits
            logps = _get_batch_logps(logits, labels, average_log_prob=True)
            losses = -logps.mean()
            
            metrics[f'loss/{train_test}'] = losses.cpu().numpy().tolist()
        
        elif loss_config.name == 'ift':
            input_ids = batch['target_input_ids']
            attention_mask = batch['target_attention_mask']
            labels = batch['target_labels']
            loss_mask = (labels[:, 1:] != -100)
            
            # self.debug_inputs(batch)
            
            inputs_embeds = self.embed_tokens(input_ids)
            
            with torch.no_grad():
                logits = self.policy(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                logps = _get_batch_logps(logits, labels, per_token_prob=True)
                losses_sft = -logps.sum(-1) / loss_mask.sum(-1)
                all_devices_losses_sft = all_gather_if_needed(losses_sft.mean().detach(), self.rank, self.world_size)
                metrics[f'loss/{train_test}_sft'] = all_devices_losses_sft.cpu().numpy().tolist()
            
            _lambda = self.update_lambda(step_idx=self.batch_counter)
            
            metrics[f'lambda/{train_test}'] = _lambda
            
            tokens_further = torch.cat((input_ids[:, 0].unsqueeze(-1), self.get_sample_function(logits)[:, :-1]), dim=-1)
            input_ids_further = torch.where(labels==-100, input_ids, tokens_further)
            attention_mask_further = attention_mask
            
            inputs_embeds_further = self.embed_tokens(input_ids_further)
            inputs_embeds_further = (1 - _lambda) * inputs_embeds + _lambda * inputs_embeds_further
            logits_further = self.policy(inputs_embeds=inputs_embeds_further, attention_mask=attention_mask_further).logits
            
            logps_further = _get_batch_logps(logits_further, labels, per_token_prob=True)
            
            with torch.no_grad():
                losses_further = -logps_further.sum(-1) / loss_mask.sum(-1)
                all_devices_losses_further = all_gather_if_needed(losses_further.mean().detach(), self.rank, self.world_size)
                metrics[f'loss/{train_test}_further'] = all_devices_losses_further.cpu().numpy().tolist()
            
            cumsum_weight = self.get_cumsum_weight(
                logps=logps_further, 
                loss_mask=loss_mask, 
                gamma=loss_config.gamma,
                propagation_type=loss_config.propagation_type, 
                propagation_norm=loss_config.propagation_norm, 
                propagation_side=loss_config.propagation_side
            )
            
            losses = ((-logps_further * cumsum_weight).sum(-1) / loss_mask.sum(-1)).mean()
    
            all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        
        elif loss_config.name == 'orpo':
            logps_chosen_sum, logps_rejected_sum = self.concatenated_forward(self.policy, batch)
            
            loss_mask_chosen = (batch['chosen_labels'][:, 1:] != -100)
            loss_mask_rejected = (batch['rejected_labels'][:, 1:] != -100)
            
            logps_chosen = logps_chosen_sum / loss_mask_chosen.sum(-1)
            logps_rejected = logps_rejected_sum / loss_mask_rejected.sum(-1)
            
            loss_chosen = -logps_chosen
            
            log_odds = (logps_chosen - logps_rejected) - (torch.log(1 - torch.exp(logps_chosen)) - torch.log(1 - torch.exp(logps_rejected)))
            sig_ratio = torch.sigmoid(log_odds)
            ratio = torch.log(sig_ratio)
            
            beta = loss_config.beta
            losses = (loss_chosen - beta * ratio).mean()
            
            all_devices_losses_sft = all_gather_if_needed(loss_chosen.mean().detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}_sft'] = all_devices_losses_sft.cpu().numpy().tolist()
            
            all_devices_loss = all_gather_if_needed(losses.mean().detach(), self.rank, self.world_size)
            metrics[f'loss/{train_test}'] = all_devices_loss.cpu().numpy().tolist()
            
        return losses, metrics
    
    def get_batch_embeddings(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the embeddings of the chosen and rejected responses for the given batch of inputs."""
        # todo: check whether the function is correct 
        with torch.no_grad():
            policy_chosen_embeddings = self.policy.get_input_embeddings()(batch['chosen_input_ids'])
            policy_rejected_embeddings = self.policy.get_input_embeddings()(batch['rejected_input_ids'])

        if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
            with torch.no_grad():
                reference_chosen_embeddings = self.reference_model.get_input_embeddings()(batch['chosen_input_ids'])
                reference_rejected_embeddings = self.reference_model.get_input_embeddings()(batch['rejected_input_ids'])
        elif self.config.loss.name == 'sft':
            reference_chosen_embeddings = torch.zeros_like(policy_chosen_embeddings)
            reference_rejected_embeddings = torch.zeros_like(policy_rejected_embeddings)
        else:
            raise ValueError(f'unknown loss {self.config.loss.name}')

        return policy_chosen_embeddings, policy_rejected_embeddings, reference_chosen_embeddings, reference_rejected_embeddings

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()
        elif self.config.loss.name == 'gpo':
            self.policy_weak.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        
        if self.config.checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=self.config.checkpoint_path)
        
        self.step_idx = self.checkpoint_example_idx if self.config.checkpoint_path is not None else 0
        
        for batch in tqdm.tqdm(self.train_loader, total=self.train_iterations, desc='Training'):
            if self.config.checkpoint_path is not None:
                if self.checkpoint_batch_idx > self.batch_counter:
                    self.batch_counter += 1
                    self.example_counter += self.config.batch_size
                    continue
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        loss, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        try:
                            all_eval_metrics[k].extend(v)
                        except:
                            all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo', 'mypo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####
            
            #### BEGIN TRAINING ####
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)
            
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, 'cpu')
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                
                loss.backward()
                
                for k, v in metrics.items():
                    try:
                        batch_metrics[k].extend(v)
                    except:
                        batch_metrics[k].append(v)
                    
            # gather the gradients
            for p in self.policy.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
                p.grad.data /= self.config.gradient_accumulation_steps
            
            grad_norm = self.clip_gradient()
            
            dist.barrier()
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            learning_rate = self.optimizer.param_groups[0]['lr']
            
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)
            batch_metrics['loss/train'].append(loss.item())
            batch_metrics['learning_rate'].append(learning_rate)
            
            rank0_print(f'batch {self.batch_counter}: loss {loss.item()}, grad_norm {grad_norm}, lr {learning_rate}, {examples_per_second} examples/s')
                
            self.batch_counter += 1
            self.example_counter += self.config.batch_size
            
            if self.config.loss.name in {'ift'}:
                self.embed_tokens.weight.data = self.policy.state_dict()['model.embed_tokens.weight'].clone().to(self.embed_tokens.weight.dtype)
                dist.barrier()
                
            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            
            self.step_idx += 1
            #### END TRAINING ####
    
    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint from disk."""
        rank0_print(f'Loading checkpoint from {checkpoint_path}')
        policy = torch.load(f'{checkpoint_path}/policy.pt', map_location='cpu')
        self.policy.load_state_dict(policy['state'])
        optimizer = torch.load(f'{checkpoint_path}/optimizer.pt', map_location='cpu')
        self.optimizer.load_state_dict(optimizer['state'])
        lr_scheduler = torch.load(f'{checkpoint_path}/scheduler.pt', map_location='cpu')
        self.lr_scheduler.load_state_dict(lr_scheduler['state'])
        
        self.checkpoint_example_idx = policy['step_idx']
        self.checkpoint_batch_idx = policy['step_idx'] // self.config.batch_size
        rank0_print(f'Loaded checkpoint from {checkpoint_path} at step {self.checkpoint_example_idx}')
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        # if self.config.loss.fuse_mode == 'embeds':
        #     self.policy.model.embed_tokens.load_state_dict(self.embed_tokens.state_dict())
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.lr_scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        
    def watch_grad(self):
        """Watch the gradient of the policy during training, but don't make parameter updates."""
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=self.num_iterations)
        
        self.grad_list = []
        self.policy.train()
        for batch in tqdm.tqdm(self.train_iterator, total=self.num_iterations):
            local_microbatch = slice_and_move_batch_for_device(batch, self.rank, self.world_size, self.rank)
            loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
            loss.backward()
            grad_norm = self.clip_gradient()
            
            
            self.grad_list.append(grad_norm)
            out_json = {"grad_norm": grad_norm,
                        "prompt": batch['prompt'][0],
                        "chosen": batch['chosen_response_only'][0],
                        "rejected": batch['rejected_response_only'][0]
                        }
            with open(f'figure/{self.config.datasets}_{self.config.model}_grad_norm.jsonl', 'a') as f:
                f.write(json.dumps(out_json) + "\n")
            with open(f'figure/{self.config.datasets}_{self.config.model}_grad_norm_indent.jsonl', 'a') as f:
                f.write(json.dumps(out_json, indent=2) + "\n")
                
            self.optimizer.zero_grad()
        
        plt.plot(list(range(1, self.num_iterations+1)), self.grad_list)
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm of the Policy')
        plt.savefig('figure/grad_norm.png')
    
    def watch_embedding(self):
        """Watch the embedding of the policy during training, but don't make parameter updates."""
        # todo: finish the function
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
        self.policy.train()
        for batch in tqdm.tqdm(self.train_iterator, total=self.num_iterations):
            local_microbatch = slice_and_move_batch_for_device(batch, self.rank, self.world_size, self.rank)


class FSDPTrainer(BasicTrainer):
    
    def __init__(self, 
                 policy: nn.Module, 
                 config: DictConfig, 
                 seed: int, 
                 run_dir: str, 
                 policy_weak: Optional[nn.Module] = None,
                 reference_model: Optional[nn.Module] = None, 
                 truncation_side: str = 'right',
                 padding_side: str = 'right',
                 rank: int = 0, 
                 world_size: int = 1
        ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy=policy, 
                         config=config, 
                         seed=seed, 
                         run_dir=run_dir, 
                         policy_weak=policy_weak, 
                         reference_model=reference_model,
                         truncation_side=truncation_side,
                         padding_side=padding_side, 
                         rank=rank, 
                         world_size=world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        assert self.config.batch_size % self.config.gradient_accumulation_steps % self.world_size == 0, 'batch_size must be divisible by gradient_accumulation_steps and world_size'
        
        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None, # if not (self.config.loss.fuse_mode == 'embeds') else [policy.model.embed_tokens],
            limit_all_gathers=False, # TODO: make sure whether the gradient accumulation is affected by this setting
            use_orig_params=False,
            sync_module_states=False
        )
        self.shared_fsdp_kwargs = shared_fsdp_kwargs
        
        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'sft', 'dpo', 'ipo'}:
            if config.loss.reference_free is False:
                rank0_print('Sharding reference model...')
                self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
            else:
                self.reference_model = None
                
            self.policy_weak = None
            
            
        print('Loaded model on rank', rank)
        dist.barrier()
        
        # have to initialize the optimizer and scheduler after __init__ to avoid conflict with FSDP
        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), self.config.lr)
         
        rank0_print(f'Using {self.config.lr_scheduler} learning rate scheduler')
        if self.config.lr_scheduler == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                                  lr_lambda=lambda step: min(1.0, (step + 1) / (self.warmup_steps + 1))
                                )
        elif self.config.lr_scheduler == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                  lr_lambda=lambda step: cosine_with_warmup(step, 
                                                                                                            num_warmup_steps=self.warmup_steps, 
                                                                                                            num_training_steps=self.train_iterations)
                                )
        else:
            raise ValueError(f'unknown lr_scheduler {self.config.lr_scheduler}')
    
    
    
    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint from disk."""
        rank0_print(f'Loading checkpoint from {self.config.checkpoint_path}')
        policy = torch.load(f'{self.config.checkpoint_path}/policy.pt', map_location='cpu')
        self.policy.load_state_dict(policy['state'])
        
        optimizer = torch.load(f'{self.config.checkpoint_path}/optimizer.pt', map_location='cpu')
        self.optimizer.load_state_dict(FSDP.optim_state_dict_to_load(self.policy, self.optimizer, optimizer['state']))
        
        lr_scheduler = torch.load(f'{self.config.checkpoint_path}/scheduler.pt', map_location='cpu')
        self.lr_scheduler.load_state_dict(lr_scheduler['state'])

        self.checkpoint_example_idx = policy['step_idx']
        self.checkpoint_batch_idx = policy['step_idx'] // self.config.batch_size
        rank0_print(f'Loaded checkpoint from {self.config.checkpoint_path} at step {self.checkpoint_example_idx}')
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.lr_scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
