model = dict(
    text_encoder=dict(
        type='PretrainedQwenImageTextEncoder',
        from_pretrained='Qwen/Qwen-Image',
        pad_seq_len=512,
    ),
)
