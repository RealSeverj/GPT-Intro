import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.transformer import MiniGPT
from src.utils.data_loader import TextDataset
from config import settings as cfg
from tqdm import tqdm
import os

def main():
    # 初始化模型
    model = MiniGPT()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.TRAIN_CONFIG["learning_rate"])

    # 数据加载配置
    dataset = TextDataset('data/raw/shakespeare.txt')
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # 设置为0以使用CPU
        pin_memory=False
    )

    # 训练循环配置
    criterion = nn.CrossEntropyLoss()
    model.train()

    # 创建输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(cfg.TRAIN_CONFIG["base_epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}')
        for batch_idx, (x, y) in progress_bar:
            # 使用view代替reshape以提高效率
            x = x.view(-1, cfg.MODEL_CONFIG["block_size"])
            y = y.view(-1, cfg.MODEL_CONFIG["block_size"])

            # 初始化memory张量
            memory = torch.zeros(x.size(0), cfg.MODEL_CONFIG["block_size"], cfg.MODEL_CONFIG["d_model"])

            # 前向传播
            outputs = model(x, memory)
            loss = criterion(
                outputs.view(-1, cfg.MODEL_CONFIG["vocab_size"]),
                y.view(-1)
            )

            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)  # 减少内存操作
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=avg_loss)

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print("训练完成！")


if __name__ == "__main__":
    main()
