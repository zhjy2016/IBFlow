# ~70GB VRAM per GPU at the default batch size.

model = dict(
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')),
)

train_cfg = dict(
    grad_accum_batch_size=4,
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)

optimizer = dict(
    diffusion=dict(
        type='AdamW8bit',
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        paramwise_cfg=dict(
            custom_keys=dict(
                proj_out_loggamma=dict(lr_mult=0.1))),
    ),
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    by_epoch=False,
)

runner = dict(
    type='DynamicIterBasedRunnerMod',
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    gc_interval=20,
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
