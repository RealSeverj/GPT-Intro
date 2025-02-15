# 模型参数
MODEL_CONFIG = {
    "vocab_size": 256,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 6,
    "block_size": 64
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 32,
    "base_epochs": 3,
    "learning_rate": 1e-4,
    "max_seq_len": 256,
    "max_iters": 1000,
    "lr_decay_iters": 1000,
    "dropout": 0.1
}
