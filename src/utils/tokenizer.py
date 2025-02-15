from config import settings as cfg


class CharTokenizer:
    def __init__(self, vocab_size=cfg.MODEL_CONFIG["vocab_size"]):
        # 创建一个包含所有ASCII字符的词汇表
        self.vocab = {chr(i): i for i in range(256)}

        # 如果需要，扩展字符集，加入更多的字符（例如中文字符）
        self.vocab.update({chr(i): i + 256 for i in range(0x4e00, 0x9fff + 1)})  # Add Chinese characters to the vocab
        # 添加常见标点符号和符号（例如，逗号，感叹号）
        punctuation = ['，', '！', '。', '？', '；', '：', '“', '”', '（', '）', '【', '】', '《', '》']
        for i, char in enumerate(punctuation, start=len(self.vocab)):
            self.vocab[char] = i

        # 处理反向词汇表
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 如果有特殊token，如 <UNK> 或 <PAD>，可以添加
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.unk_token_id = self.vocab.get(self.unk_token, self.vocab_size)  # 默认插入新的 token
        self.pad_token_id = self.vocab.get(self.pad_token, self.vocab_size + 1)  # 默认插入新的 token

        # 确保词汇表的大小与模型一致
        self.vocab_size = vocab_size

    def encode(self, text):
        return [self.vocab.get(c, self.unk_token_id) for c in text]  # 使用 <UNK> 替代词汇表外的字符

    def decode(self, tokens):
        # 遇到超出词汇表的 token 时，映射为 <UNK>，避免乱码
        return ''.join([self.inv_vocab.get(t, self.unk_token) for t in tokens])
