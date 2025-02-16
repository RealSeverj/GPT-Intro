import os
import sys
import torch.multiprocessing as mp
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # 新增混合精度支持
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.transformer import MiniGPT
from src.utils.data_loader import TextDataset, collate_fn
from config import settings as cfg
from tqdm import tqdm

# 设置 spawn 启动方式
mp.set_start_method('spawn', force=True)

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 启用CUDA加速
    print(f"Using device: {device}")

    # 初始化模型
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG["learning_rate"])
    scaler = GradScaler() if device.type == "cuda" else None  # 混合精度梯度缩放器

    # 数据加载优化配置
    dataset = TextDataset(cfg.TRAIN_CONFIG["dataset"], cache_dir='data', device=device)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=12 if device.type == "cuda" else 16,  # GPU建议8-12个worker
        persistent_workers=True,  # 保持worker进程
        collate_fn=collate_fn
    )

    # 训练配置
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    # 创建输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model is on device: {next(model.parameters()).device}")

    for epoch in range(cfg.TRAIN_CONFIG["base_epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}')

        for batch_idx, (x, y) in progress_bar:
            # 数据预处理
            if x.size(1) % cfg.MODEL_CONFIG["block_size"] != 0 or y.size(1) % cfg.MODEL_CONFIG["block_size"] != 0:
                continue

            x = x.view(-1, cfg.MODEL_CONFIG["block_size"]).to(device, non_blocking=True)
            y = y.view(-1, cfg.MODEL_CONFIG["block_size"]).to(device, non_blocking=True)

            # 使用混合精度训练
            with autocast(enabled=device.type == "cuda"):
                memory = torch.zeros(x.size(0), cfg.MODEL_CONFIG["block_size"], cfg.MODEL_CONFIG["d_model"],
                                     device=device)
                outputs = model(x, memory)
                loss = criterion(outputs.view(-1, cfg.MODEL_CONFIG["vocab_size"]), y.view(-1))

            # 反向传播优化
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()  # 缩放梯度
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 内存管理
            del memory, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 记录损失
            total_loss += loss.item()
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
    print("Training complete!")


if __name__ == "__main__":
    main()