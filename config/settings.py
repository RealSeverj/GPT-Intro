# 模型参数
MODEL_CONFIG = {
    "vocab_size": 66000,
    "d_model": 256,
    "nhead": 2,
    "num_layers": 2,
    "block_size": 64
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 32,
    "base_epochs": 1,
    "learning_rate": 1e-4,
    "max_seq_len": 256,
    "max_iters": 500,
    "lr_decay_iters": 500,
    "dropout": 0.1
}
