import torch
import torch.nn as nn
from config import settings as cfg

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化词嵌入层
        self.embed = nn.Embedding(cfg.MODEL_CONFIG["vocab_size"],
                                  cfg.MODEL_CONFIG["d_model"])
        # 初始化位置嵌入层
        self.pos_embed = nn.Embedding(cfg.MODEL_CONFIG["block_size"],
                                      cfg.MODEL_CONFIG["d_model"])
        # 初始化Transformer解码器
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.MODEL_CONFIG["d_model"],
                nhead=cfg.MODEL_CONFIG["nhead"],
                dim_feedforward=cfg.MODEL_CONFIG["d_model"] * 4,
                batch_first=True
            ),
            num_layers=cfg.MODEL_CONFIG["num_layers"]
        )
        # 初始化全连接层
        self.fc = nn.Linear(cfg.MODEL_CONFIG["d_model"],
                            cfg.MODEL_CONFIG["vocab_size"])

        # 预生成位置索引
        self.register_buffer("position_ids",
                             torch.arange(cfg.MODEL_CONFIG["block_size"]))

    def forward(self, x, memory):
        # 获取序列长度
        seq_len = x.size(1)
        # 获取词嵌入
        token_emb = self.embed(x)
        # 获取位置嵌入
        pos_emb = self.pos_embed(self.position_ids[:seq_len])
        # 将词嵌入和位置嵌入相加
        x = token_emb + pos_emb

        # 生成注意力掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # 通过Transformer解码器进行处理
        x = self.transformer(x, memory=memory, tgt_mask=tgt_mask)
        # 通过全连接层进行输出
        return self.fc(x)
