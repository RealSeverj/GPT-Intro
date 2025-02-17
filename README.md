<h1 style="display:flex;justify-content: center">GPT-Intro</h1>

<p style="display:flex;justify-content: right">By Severj</p>

<h2>运行效果</h2>
<img src="data/assets/hello world.jpg" alt="">

<h2>快速上手</h2>

安装依赖：
```
pip install -r requirements.txt
```
运行训练：
```
python scripts/train.py
```
若无法连接到Huggingface服务器，可以使用以下命令运行训练：
``` 
HF_ENDPOINT=https://hf-mirror.com python ./scripts/train.py
```
训练结果对话：
```
python scripts/test.py
```
超参数 (Hyperparameters) 可以在 `config/settings.py` 中修改。

目前数据集大小为1GB，在线GPU（Nvidia A10）训练时间约为1小时。
