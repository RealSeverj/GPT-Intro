# 模型参数
MODEL_CONFIG = {
    "vocab_size": 256,
    "d_model": 256,
    "nhead": 2,
    "num_layers": 4,
    "block_size": 256
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 256,
    "base_epochs": 3,
    "learning_rate": 5e-5,
    "max_seq_len": 256,
    "max_iters": 500,
    "lr_decay_iters": 500,
    "dropout": 0.1,
    "dataset": "roneneldan/TinyStories"

}
