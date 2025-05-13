# fine-tune-zero-shot-classifier

## 项目简介
本项目基于 Huggingface Transformers，针对零样本文本分类任务，支持自定义数据集微调、模型评估与推理服务。适用于小样本、标签灵活的文本分类场景。

## 主要功能
- 支持自定义数据集的多分类模型微调（以BART-large-MNLI为例）
- 自动划分训练/验证集，支持Early Stopping防止过拟合
- 训练日志与模型断点自动保存
- 推理服务支持FastAPI，自动生成Swagger文档
- 支持单条与批量文本推理

## 依赖环境
- Python >= 3.9
- torch >= 1.10
- transformers >= 4.30
- datasets
- scikit-learn
- pandas
- tqdm
- fastapi
- uvicorn
- pydantic

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据格式说明
训练数据需为Excel文件（如`data/FileExtractorData.xlsx`），包含至少两列：
- `paragraph`：待分类文本
- `label`：标签（整数，范围0~3）

## 训练模型
在项目根目录下运行：
```bash
python src/train.py
```
训练过程会自动保存最优模型到`models/final_model`，并在`models/checkpoints/`下保存断点。

## 查看模型参数量
```bash
# 以某个checkpoint为例
python -c "from transformers import AutoModelForSequenceClassification; model=AutoModelForSequenceClassification.from_pretrained('./models/checkpoints/checkpoint-68'); print(sum(p.numel() for p in model.parameters()))"
```

## 启动推理服务（FastAPI）
1. 启动服务：
```bash
uvicorn src.inference:app --host 0.0.0.0 --port 5000
```
2. 访问Swagger文档：
```
http://localhost:5000/docs
```
3. 调用接口示例：
```bash
curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d '{"text": "测试文本内容"}'
```

## 常见问题与注意事项
- **标签范围**：仅支持0、1、2、3四类，需保证数据标签合法。
- **显存不足**：可适当减小batch size、max_length。
- **JSON转义**：推理接口传参时，文本中如有反斜杠需写成`\\`。
- **多卡训练**：Trainer自动支持，无需手动DataParallel。
- **模型路径**：推理和参数统计时请确保路径正确。

## 目录结构
```
├── baseline/                # 预训练模型及相关资源
├── data/                    # 数据集文件夹
│   └── FileExtractorData.xlsx
├── models/                  # 训练输出模型与断点
│   ├── checkpoints/
│   └── final_model/
├── src/
│   ├── train.py             # 训练主程序
│   ├── inference.py         # 推理服务（FastAPI）
│   └── ...
├── requirements.txt         # 依赖包列表
└── README.md                # 项目说明
```
