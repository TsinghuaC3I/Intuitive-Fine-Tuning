# Equivalent to SFT, but will be slower than setting loss=sft and loss.name=sft
python -u \
    src/train.py \
    model=mistral7b \
    datasets=[ultrachat,ultrafeedback] \
    exp_name=ultra_ift_mistral7b_fsdp \
    loss=ift \
    loss.name=ift \
    loss.gamma=0.00 \
    loss.min_lambda=0.0 \
    loss.max_lambda=0.0 \
    n_epochs=3 \
    n_examples=61136 \
    batch_size=512 \
    gradient_accumulation_steps=64 \
    eval_batch_size=32 \
    lr=5e-7 \
    warmup_ratio=0.15 \
    max_prompt_length=1024 \
    max_length=1024 \
    trainer=FSDPTrainer \
    optimizer=RMSprop \
    lr_scheduler=cosine

# IFT
python -u \
    src/train.py \
    model=mistral7b \
    datasets=[ultrachat,ultrafeedback] \
    exp_name=ultra_ift_mistral7b_fsdp \
    loss=ift \
    loss.name=ift \
    loss.gamma=0.95 \
    loss.min_lambda=0.2 \
    loss.max_lambda=0.2 \
    loss.propagation_type=loss \
    loss.propagation_norm=L1 \
    loss.propagation_side=left \
    n_epochs=3 \
    n_examples=61136 \
    batch_size=512 \
    gradient_accumulation_steps=64 \
    eval_batch_size=32 \
    lr=5e-7 \
    warmup_ratio=0.15 \
    max_prompt_length=1024 \
    max_length=1024 \
    trainer=FSDPTrainer \
    optimizer=RMSprop \
    lr_scheduler=cosine

# IFT with trained checkpoint
python -u \
    src/train.py \
    model=mistral7b \
    datasets=[ultrachat,ultrafeedback] \
    exp_name=ultra_ift_mistral7b_fsdp \
    loss=ift \
    loss.name=ift \
    loss.gamma=0.95 \
    loss.min_lambda=0.2 \
    loss.max_lambda=0.2 \
    loss.propagation_type=loss \
    loss.propagation_norm=L1 \
    loss.propagation_side=left \
    n_epochs=3 \
    n_examples=61136 \
    batch_size=512 \
    gradient_accumulation_steps=64 \
    eval_batch_size=32 \
    lr=5e-7 \
    warmup_ratio=0.15 \
    max_prompt_length=1024 \
    max_length=1024 \
    trainer=FSDPTrainer \
    optimizer=RMSprop \
    lr_scheduler=cosine \
    checkpoint_path=path/to/checkpoint
