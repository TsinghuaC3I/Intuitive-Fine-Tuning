import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
import peft
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource

# import deepspeed
import argparse

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, policy_weak: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    # elif 'DeepSpeed' in config.trainer:
    #     deepspeed.init_distributed(dist_backend="nccl", rank=rank, world_size=world_size)#, distributed_port=config.deepspeed_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, policy_weak=policy_weak, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port
    elif 'DeepSpeed' in config.trainer and config.deepspeed_port is None:
        free_port = get_open_port()
        print('no DeepSpeed port specified; using open port for DeepSpeed:', free_port)
        config.deepspeed_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, 
        cache_dir=get_local_dir(config.local_dirs), 
        low_cpu_mem_usage=True, 
        torch_dtype=policy_dtype, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2" if config.flash_attn else None,
        **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'sft', 'dpo', 'ipo', 'kto', 'orpo'} and config.loss.reference_free is False:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, trust_remote_code=True, **model_kwargs)
        disable_dropout(reference_model)
        
        policy_weak = None
    elif config.loss.name == 'gpo': # TODO: for debug, can be simplified
        # equal to rejected_from == 'weak', but using more computational resources
        assert not (config.loss.weak_free and config.loss.rejected_from == 'reference')
        # then we will lose the reject sample source
        assert not (config.loss.weak_free and config.loss.reference_free)
        # TODO: the logic need to be checked
        if config.loss.weak_free or (config.loss.reference_free is False) or (config.loss.kl_free is False) or (config.loss.rejected_from == 'reference'):
            print('building reference model')
            reference_model_dtype = getattr(torch, config.model.reference_dtype)
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path, 
                cache_dir=get_local_dir(config.local_dirs), 
                low_cpu_mem_usage=True, 
                torch_dtype=reference_model_dtype, 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2" if config.flash_attn else None,
                **model_kwargs)
            disable_dropout(reference_model)
        else:
            reference_model = None
        
        if config.loss.weak_free is False:
            print('building weak policy')
            policy_weak_dtype = getattr(torch, config.model.policy_dtype)
            policy_weak = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path, 
                cache_dir=get_local_dir(config.local_dirs), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_weak_dtype, 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2" if config.flash_attn else None,
                **model_kwargs)
            disable_dropout(policy_weak)
        else:
            policy_weak = None
        
    else:
        policy_weak = None
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo', 'mypo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    if config.model.lora is True:
        loftq_config = peft.LoftQConfig(loftq_bits=4, 
                                        loftq_iter=1)
        lora_config = peft.LoraConfig(init_lora_weights="loftq", 
                                      loftq_config=loftq_config)
        policy = peft.get_peft_model(policy, lora_config)
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, policy_weak, reference_model), join=True)
    elif 'DeepSpeed' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for DeepSpeed training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        worker_main(config.local_rank, world_size, config, policy, reference_model)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, policy_weak, reference_model)

if __name__ == '__main__':
    main()
    