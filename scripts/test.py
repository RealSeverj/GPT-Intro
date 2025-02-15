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

def generate_text(instruction, model, tokenizer, max_length=200, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(instruction)
    input_ids = [i for i in input_ids if i < tokenizer.vocab_size]  # Ensure indices are within vocab size
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(input_ids[-cfg.MODEL_CONFIG["block_size"]:]).unsqueeze(0)
            outputs = model(inputs, memory=torch.zeros(1, cfg.MODEL_CONFIG["block_size"], cfg.MODEL_CONFIG["d_model"]))
            logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_filtering(logits, top_k=top_k)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            if next_id >= tokenizer.vocab_size:
                next_id = 0  # Handle out-of-vocab indices
            input_ids.append(next_id)
            if next_id == tokenizer.encode('\n')[0]:  # Assuming newline character indicates end of response
                break
    return tokenizer.decode(input_ids)

def top_k_filtering(logits, top_k=50):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[-1].unsqueeze(0)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    return logits

if __name__ == "__main__":
    tokenizer = CharTokenizer()
    model = load_model("output/final_model.pth")

    while True:
        instruction = input("输入指令（输入q退出）>> ")
        if instruction.lower() == 'q':
            break
        generated = generate_text(instruction, model, tokenizer)
        print("\n生成结果：")
        print(generated + "\n")
