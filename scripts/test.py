import torch
import torch.nn.functional as F
from src.model import MiniGPT
from src.utils.tokenizer import CharTokenizer
from config import settings as cfg

def load_model(model_path):
    model = MiniGPT()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(prompt, model, tokenizer, max_length=200, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(input_ids[-cfg.MODEL_CONFIG["block_size"]:]).unsqueeze(0)
            outputs = model(inputs, memory=torch.zeros(1, cfg.MODEL_CONFIG["block_size"], cfg.MODEL_CONFIG["d_model"]))
            logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_filtering(logits, top_k=top_k)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            input_ids.append(next_id)
            if next_id == tokenizer.encode('\n')[0]:  # 假设换行符表示回答结束
                break
    return tokenizer.decode(input_ids)

def top_k_filtering(logits, top_k=50):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[-1].unsqueeze(0)
        # noinspection PyTypeChecker
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    return logits


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    model = load_model("output/final_model.pth")

    while True:
        prompt = input("输入起始文本（输入q退出）>> ")
        if prompt.lower() == 'q':
            break
        generated = generate_text(prompt, model, tokenizer)
        print("\n生成结果：")
        print(generated + "\n")
