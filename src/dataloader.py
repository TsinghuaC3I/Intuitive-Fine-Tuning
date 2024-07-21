import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np

import tqdm
import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils import rank0_print, get_local_dir, TemporarilySeededRandom
import time
import math

@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset. If you want each prompt to be uniquely associated with an Example instance, save it in a dict.
    """
    prompt: str = ''                                            # prompt text
    chosen: str = ''                                            # the chosen text
    rejected: str = ''                                          # the rejected text                        # if truncation needed, keep the beginning (keep_start) or end (keep_end) (only override default for SHP)                            # the unformatted prompt (needed to recover instruction for AlpacaEval)
        
class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, 
                 name,
                 truncation_mode: str = 'keep_start'):
        self.name = name
        self.data = defaultdict(Example)
        self.truncation_mode = truncation_mode

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("key must be a string")
        
        if not isinstance(value, Example):
            raise ValueError("value must be a Example")
        
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)

def get_orca(split, preprocess, silent, cache_dir, n_examples=None) -> Dataset:
    dataset = datasets.load_dataset('/root/pubdatasets/orca_dpo_pairs', split=split, cache_dir=cache_dir)

    data = Dataset(name='orca', truncation_mode='keep_start')
    count = 0
    for row in tqdm.tqdm(dataset, desc='Processing Orca', disable=silent):
        title = row['prompt']
        
        prompt = row['chosen'][:-1]
        if prompt[0]['role'] != 'system':
            prompt.insert(0, {"role": "system", "content": ""})
            
        chosen = row['chosen'][-1:]
        rejected = row['rejected'][-1:]
        
        prompt, chosen, rejected = preprocess(prompt, chosen, rejected, truncation_mode='keep_start')
        
        data[title].prompt = prompt
        data[title].chosen = chosen
        data[title].rejected = rejected
        
        count += 1
        if n_examples is not None and count >= n_examples:
            break
    
    return data

def get_ultrafeedback(split, preprocess, silent, cache_dir, n_examples=None) -> Dataset:
    split += "_prefs"
    
    dataset = datasets.load_dataset('/root/pubdatasets/UltraFeedback/others/HuggingFaceH4/ultrafeedback_binarized', split=split, cache_dir=cache_dir)

    data = Dataset(name='ultrafeedback', truncation_mode='keep_start')
    count = 0
    for row in tqdm.tqdm(dataset, desc='Processing UltraFeedback', disable=silent):
        title = row['prompt']
        
        prompt = row['chosen'][:-1]
        if prompt[0]['role'] != 'system':
            prompt.insert(0, {"role": "system", "content": ""})
        chosen = row['chosen'][-1:]
        rejected = row['rejected'][-1:]
        
        # prompt, chosen, rejected = preprocess(prompt, chosen, rejected, truncation_mode='keep_start')
        data[title].type = "pairwise"
        data[title].prompt = prompt
        data[title].chosen = chosen
        data[title].rejected = rejected
        
        count += 1
        if n_examples is not None and count >= n_examples:
            break
    
    return data

def get_ultrachat(split, preprocess, silent, cache_dir, n_examples=None) -> Dataset:
    split += "_sft"
    
    dataset = datasets.load_dataset('/root/pubdatasets/UltraChat/others/HuggingFaceH4/ultrachat_200k', split=split, cache_dir=cache_dir)

    data = Dataset(name='ultrachat', truncation_mode='keep_start')
    count = 0
    for row in tqdm.tqdm(dataset, desc='Processing UltraChat', disable=silent):
        title = row['prompt']
        
        prompt = row['messages'][:-1]
        if prompt[0]['role'] != 'system':
            prompt.insert(0, {"content": "", "role": "system"})
        messages = row['messages'][-1:]
        
        
        # prompt, messages = preprocess(prompt, messages, truncation_mode='keep_start')
        data[title].type = "single"
        data[title].prompt = prompt
        data[title].chosen = messages
        data[title].rejected = messages
        
        count += 1
        if n_examples is not None and count >= n_examples:
            break
        
    return data

def get_orcamath(split, preprocess, silent, cache_dir, n_examples=None) -> Dataset:
    # dataset = datasets.load_dataset('microsoft/orca-math-word-problems-200k', split=split, cache_dir=cache_dir)
    dataset = datasets.load_dataset('parquet', 
                                    data_dir='/root/pubdatasets/orca-math/microsoft/orca-math-word-problems-200k/data', 
                                    data_files={'train': 'train-00000-of-00001.parquet'},
                                    cache_dir=cache_dir)
    dataset = dataset['train']
    
    if split == 'test':
        dataset = [{'question': q, 'answer': a} for q, a in zip(dataset[:1000]['question'], dataset[:1000]['answer'])]
    
    data = Dataset(name='orcamath', truncation_mode='keep_start')
    count = 0
    for row in tqdm.tqdm(dataset, desc='Processing OrcaMath', disable=silent):
        title = row['question']
        
        prompt = [{"role": "system", "content": ""},
                  {"role": "user", "content": row['question']}]
        
        messages = [{"role": "assistant", "content": row['answer']}]
        
        prompt, chosen, rejected = preprocess(prompt, chosen, rejected, truncation_mode='keep_start')
        
        data[title].prompt = prompt
        data[title].chosen = messages
        data[title].rejected = messages
        
        count += 1
        if n_examples is not None and count >= n_examples:
            break
    
    return data

class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batcch elements will be different depending
    on whether you're doing SFT, aligning with a pairwise loss like DPO, or alignment with a unary loss like KTO. 
    """
    def __init__(self, 
                 names: List[str],      # e.g., ['shp', 'oasst']; should have  get_{name} method in this file
                 tokenizer,                     # Huggingface tokenizer object
                 split: str = 'train',
                 loss_name: str = 'dpo',
                 batch_size: int = 1,
                 max_length: int = 512,         # max length of prompt + response
                 max_prompt_length: int = 128,  # max length of prompt alone
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = True,
                 silent: bool = False,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        
        torch.manual_seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        if self.tokenizer.__class__.__name__ == 'Qwen2TokenizerFast':
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        self.build_assistant_starter(tokenizer=tokenizer)
        
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.seed = seed
        self.shuffle = shuffle
        self.silent = silent
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.names = names
        self.full_data = self.flatten_data() #*debug
        
        self.truncation_mode = "keep_start"# dataset.truncation_mode
    
    def build_assistant_starter(self, tokenizer):
        message = [{"content": "#"*10, "role": "assistant"}]
        starter_text = tokenizer.apply_chat_template(message, tokenize=False)
        position = starter_text.find("#"*10)

        text = starter_text[:position]
        ids = tokenizer(starter_text[:position])['input_ids']
        length = len(ids)

        self.assistant_starter = {
            "text": text,
            "ids": ids,
            "length": length}
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate function for the dataloader. 
        """
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    
    def flatten_data(self):
        """
        Flatten the data into a list of examples. 
        """
        flat_data = []
        with TemporarilySeededRandom(self.seed):
            for name in self.names:
                dataset = globals()[f"get_{name}"](self.split, self.preprocess, self.silent, self.cache_dir, self.n_examples)
                for prompt, example in dataset.data.items():
                    flat_data.append(example)
        return flat_data
            
    
    def __iter__(self):
        """
        """
        with TemporarilySeededRandom(self.seed):
            permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
            # flat_data = self.flatten_data() #*debug
        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            
            if self.shuffle:
                with TemporarilySeededRandom(next(permutation_seeds)):
                    random.shuffle(self.full_data) # otherwise, will be frontloaded with prompts in same domain
                    # random.shuffle(flat_data) #*debug
            
            batch = []
            for example in self.full_data: 
            # for example in flat_data: #*debug
                batch_element = self.get_element(example)
                if batch_element is not None:
                    batch.append(batch_element)
                    example_idx += 1
                if len(batch) == self.batch_size:
                    # example_idx += len(batch)
                    yield self.collate_fn(batch)
                    batch = []

                    if self.split != "train" and self.n_examples is not None:
                        if example_idx >= self.n_examples * len(self.names):  
                            rank0_print(f'Finished generating {self.n_examples * len(self.names)} examples on {self.split} split')
                            done = True
                            break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break
    
    def __len__(self):
        """
        The length of the dataloader. 
        """
        if self.n_examples is not None:
            if self.split == 'train':
                return math.ceil(self.n_examples * len(self.names) / self.batch_size) * self.n_epochs
            else:
                return math.ceil(self.n_examples * len(self.names) / self.batch_size) 
        else:
            return math.ceil(len(self.full_data) / self.batch_size) * self.n_epochs
            # return math.ceil(self.total_length / self.batch_size) * self.n_epochs #*debug
    
    def get_element(self, example: Example) -> Dict:
        """
        Get a single batch element. 
        """
        raise NotImplementedError
    
    def preprocess(self, *args, **kwargs):
        """
        Preprocess the prompt. 
        """
        raise NotImplementedError
    
    def tokenize_batch_element(self, 
                               prompt: str, 
                               generation: str, 
                               prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        
        messages = prompt + generation
        conversation = self.tokenizer.apply_chat_template(messages, tokenize=False)
        all_input_ids = self.tokenizer(
            conversation,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = all_input_ids.input_ids[0]
        attention_mask = all_input_ids.attention_mask[0]
        target = input_ids.clone()
        
        cur_len = 0
        for item in messages:
            tokens = self.tokenizer.apply_chat_template([item])
            
            if item['role'] != 'assistant':
                next_len = min(cur_len+len(tokens), len(target))
            else:
                next_len = min(cur_len+self.assistant_starter["length"], len(target))
                # tokens.append(-100) # TODO: add the \n token. This is for Mistral specifically
            
            target[cur_len:next_len] = torch.ones(next_len-cur_len) * -100
            
            cur_len += len(tokens)
            if cur_len >= len(target):
                break
        
        if cur_len < len(target):
            target[cur_len:] = torch.ones(len(target)-cur_len) * -100
            
        # if True:
        #     rank0_print("#"*10+" input_ids "+"#"*10)
        #     rank0_print(f"{self.tokenizer.decode(input_ids)}\n")
        #     # rank0_print([f"{self.tokenizer.decode(input_ids)}\n"])
        #     rank0_print("#"*10+" labels "+"#"*10)
        #     rank0_print(f"{self.tokenizer.decode(torch.where(target==-100, self.tokenizer.pad_token_id, target))}\n")
        #     rank0_print("#"*50)
        #     exit()
        
        # if all of the tokens are masked, return None
        # it is possible that the first user prompt is too long
        if torch.all(target == -100):
            return None
        
        return {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_labels": target,
            f"{prefix}_attention_mask": attention_mask,
        }
    
    def get_batch_element(self, example: Example) -> Dict:
        """
        Get a single batch element. 
        """
        raise NotImplementedError
    
class SFTDataLoader(DataLoader):
    """
    A data loader for SFT. 
    """
    def get_element(self, example: Example) -> Dict:
        """
        Get a single batch element. 
        """
        batch_element = self.tokenize_batch_element(
            example.prompt,
            example.chosen,
            prefix='target'
        )
        
        return batch_element
    
    def preprocess(self, prompt, generation, truncation_mode='keep_start'):
        """
        Preprocess the prompt. 
        """
        return prompt, generation
    

    def tokenize_batch_element(self, 
                               prompt: str, 
                               generation: str,
                               prefix: str='target') -> Dict:
        
        return super().tokenize_batch_element(prompt, generation, prefix=prefix)

class IFTDataLoader(SFTDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_batch_element(self, 
                               prompt: str, 
                               generation: str,
                               prefix: str="target") -> Dict:
        
        return super().tokenize_batch_element(prompt, generation)
      
class ORPODataLoader(DataLoader):
    def get_element(self, example: Example) -> Dict:
        chosen_element = self.tokenize_batch_element(
            example.prompt,
            example.chosen,
            prefix='chosen'
        )
        rejected_element = self.tokenize_batch_element(
            example.prompt,
            example.rejected,
            prefix='rejected'
        )
        # 拼接两个字典
        if chosen_element is None or rejected_element is None:
            batch_element = None
        else:
            batch_element = {**chosen_element, **rejected_element}
        
        return batch_element
    
    def preprocess(self, prompt, chosen, rejected, truncation_mode='keep_start'):
        return prompt, chosen, rejected

class KTODataLoader(DataLoader):
    def __len__(self):
        return super().__len__()
    def get_element(self, example: Example) -> Dict:
        batch_element = self.tokenize_batch_element(
            example.prompt,
            example.chosen,
            prefix='chosen'
        )
            
        return batch_element
    
    def preprocess(self, prompt, chosen, rejected, truncation_mode='keep_start'):
        return prompt, chosen, rejected
    
    def flatten_data(self):
        flat_data = []
        with TemporarilySeededRandom(self.seed):
            for name in self.names:
                dataset = globals()[f"get_{name}"](self.split, self.preprocess, self.silent, self.cache_dir, self.n_examples)
                for prompt, example in dataset.data.items():
                    if example.type == "single":
                        flat_data.append((example, 'chosen'))
                    else:
                        flat_data.append((example, 'chosen'))
                        flat_data.append((Example(prompt=example.prompt, chosen=example.rejected, rejected=example.chosen), 'rejected'))
                    
        return flat_data
    
    def __iter__(self):
        with TemporarilySeededRandom(self.seed):
            permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        
        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            
            if self.shuffle:
                with TemporarilySeededRandom(next(permutation_seeds)):
                    random.shuffle(self.full_data) # otherwise, will be frontloaded with prompts in same domain
                    # random.shuffle(flat_data) #*debug
            
            batch = []
            example_queue = []
            for example, status in self.full_data: 
                batch_element = self.get_element(example)
                
                if batch_element is not None:
                    batch_element['status'] = status
                    example_queue.append(example)
                    batch.append(batch_element)        
                    example_idx += 1
                    
                if len(batch) >= self.batch_size:
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(self.batch_size):
                        batch[i].update(self.tokenize_batch_element(
                            example_queue[i].prompt,
                            example_queue[indices[i]].chosen,
                            prefix='rejected'
                        ))
                    example_queue = []
                    
                    yield self.collate_fn(batch[:self.batch_size])
                    batch = []
                    
                    if self.split != "train" and self.n_examples is not None:
                        if example_idx >= self.n_examples * len(self.names):  
                            rank0_print(f'Finished generating {self.n_examples * len(self.names)} examples on {self.split} split')
                            done = True
                            break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

class DPODataLoader(DataLoader):
    """
    A data loader for DPO. 
    """
    def get_element(self, example: Example) -> Dict:
        """
        Get a single batch element. 
        """
        batch_element = self.tokenize_batch_element(
            example.prompt,
            example.chosen,
            example.rejected,
        )
        
        return batch_element
    
    def tokenize_batch_element(self,
                               prompt: str,
                               chosen: str,
                               rejected: str,) -> Dict:
        """
        """
        
        prompt_messages = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        chosen_messages = self.tokenizer.apply_chat_template(chosen, tokenize=False)
        rejected_messages = self.tokenizer.apply_chat_template(rejected, tokenize=False)
        
        prompt_tokens = self.tokenizer(prompt_messages, add_special_tokens=False) # added in 02.29
        chosen_tokens = self.tokenizer(chosen_messages, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected_messages, add_special_tokens=False)
        
        longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
        
        # if combined sequence is too long, first truncate prompt
        if (len(prompt_tokens['input_ids']) + longer_response_length > self.max_length) and (len(prompt_tokens['input_ids']) > self.max_prompt_length):
            if self.truncation_mode == 'keep_start':
                prompt_tokens = {k: v[:self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == 'keep_end':
                prompt_tokens = {k: v[-self.max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {self.truncation_mode}')
            
        if (len(prompt_tokens['input_ids']) + longer_response_length > self.max_length):
            response_length = self.max_length - self.max_prompt_length # TODO: check
            chosen_tokens = {k: v[:response_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[:response_length] for k, v in rejected_tokens.items()}
            
        batch_element = {}
        
        batch_element.update({f'chosen_{k}': prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens})
        batch_element[f'chosen_labels'] = batch_element[f'chosen_input_ids'][:]
        batch_element[f'chosen_labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        
        batch_element.update({f'rejected_{k}': prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens})
        batch_element[f'rejected_labels'] = batch_element[f'rejected_input_ids'][:]
        batch_element[f'rejected_labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        
        return batch_element
    
    def preprocess(self, prompt, chosen, rejected, truncation_mode='keep_start'):
        """
        Preprocess the prompt. 
        """
        
        return prompt, chosen, rejected
    
class AnalyseDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_element(self, example: Example) -> Dict:
        """
        Get a single batch element. 
        """
        batch_element = self.tokenize_batch_element(
            example.prompt,
            example.chosen,
            example.rejected,
        )
        
        return batch_element
    
    def preprocess(self, prompt, chosen, rejected, truncation_mode='keep_start'):
        """
        Preprocess the prompt. 
        """
        return prompt, chosen, rejected
    
    def tokenize_batch_element(self, prompt: str, chosen: str, rejected: str) -> Dict:
        
        prompt_messages = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        chosen_messages = self.tokenizer.apply_chat_template(chosen, tokenize=False)
        rejected_messages = self.tokenizer.apply_chat_template(rejected, tokenize=False)
        
        prompt_tokens = self.tokenizer(
            prompt_messages,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False
        )
        max_response_length = self.max_length - len(prompt_tokens['input_ids'])
        chosen_tokens = self.tokenizer(
            chosen_messages,
            max_length=max_response_length,
            truncation=True,
            add_special_tokens=False
        )
        rejected_tokens = self.tokenizer(
            rejected_messages, 
            max_length=max_response_length,
            truncation=True,
            add_special_tokens=False
        )
        
        return dict(
            prompt_input_ids=prompt_tokens['input_ids'],
            chosen_input_ids=chosen_tokens['input_ids'],
            rejected_input_ids=rejected_tokens['input_ids'],
        )
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from utils import slice_and_move_batch_for_device
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/pubmodels/transformers/chat-models/mistral-7b-sft-beta", 
        cache_dir=".cache/", 
        truncation_side='right',
        padding_side='right')
    
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    loader = SFTDataLoader(
        names=['ultrachat','ultrafeedback'],
        tokenizer=tokenizer,
        split='train',
        batch_size=16,
        max_length=2048,
        max_prompt_length=2048,
        n_epochs=3,
        seed=0,
        shuffle=False,
        silent=False,
        cache_dir=".cache/"
    )

    gradient_accumulation_steps = 2
    world_size = 8
    
    count = 0
    start_time = time.time()
    check_time = start_time
    for batch in tqdm.tqdm(loader):
        if count % 100 == 0:
            tmp_time = time.time()
            print(f"Time: {tmp_time - start_time}")
            check_time = tmp_time
        count += 1
        
    end_time = time.time()
    print(f"Time: {end_time - start_time}")