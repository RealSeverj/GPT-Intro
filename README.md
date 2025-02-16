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
