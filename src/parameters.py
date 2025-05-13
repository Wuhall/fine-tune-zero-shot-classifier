import torch
from transformers import AutoModelForSequenceClassification

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")

if __name__ == "__main__":
    model_path = "models/checkpoints/checkpoint-68"  
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    count_parameters(model)
