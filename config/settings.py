# 模型参数
MODEL_CONFIG = {
    "vocab_size": 16,
    "d_model": 16,
    "nhead": 1,
    "num_layers": 1,
    "block_size": 16
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 1,
    "base_epochs": 1,
    "learning_rate": 1e-4,
    "max_seq_len": 16,
    "max_iters": 10,
    "lr_decay_iters": 10,
    "dropout": 0.0
}
