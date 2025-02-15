from config import settings as cfg
import numpy as np


class CharTokenizer:
    def __init__(self):
        self.vocab_size = cfg.MODEL_CONFIG["vocab_size"]
        # 预生成转换表
        self.encode_table = np.zeros(256, dtype=np.int32)
        for i in range(256):
            self.encode_table[i] = i if i < self.vocab_size else 0

    def encode(self, text):
        """使用numpy进行批量编码"""
        arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
        return self.encode_table[arr].tolist()

    def decode(self, token_ids):
        """批量解码优化"""
        arr = np.array(token_ids, dtype=np.uint8)
        return bytes(arr[arr < self.vocab_size]).decode('utf-8', 'ignore')
