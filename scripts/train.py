from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.transformer import MiniGPT
from src.utils.data_loader import TextDataset, collate_fn
from config import settings as cfg
from tqdm import tqdm
import os

def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Initialize model
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG["learning_rate"])

    # Data loading configuration
    dataset = TextDataset('CausalLM/Refined-Anime-Text', cache_dir='data')
    train_loader = DataLoader(dataset, batch_size=cfg.TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=16, pin_memory=False, collate_fn=collate_fn)

    # Training loop configuration
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(cfg.TRAIN_CONFIG["base_epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}')
        for batch_idx, (x, y) in progress_bar:
            if x.size(1) % cfg.MODEL_CONFIG["block_size"] != 0 or y.size(1) % cfg.MODEL_CONFIG["block_size"] != 0:
                continue
            x = x.view(-1, cfg.MODEL_CONFIG["block_size"]).to(device)
            y = y.view(-1, cfg.MODEL_CONFIG["block_size"]).to(device)

            memory = torch.zeros(x.size(0), cfg.MODEL_CONFIG["block_size"], cfg.MODEL_CONFIG["d_model"]).to(device)

            outputs = model(x, memory)
            loss = criterion(outputs.view(-1, cfg.MODEL_CONFIG["vocab_size"]), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=avg_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print("Training complete!")

if __name__ == "__main__":
    main()