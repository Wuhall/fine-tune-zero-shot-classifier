import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def get_device():
    """获取可用的设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_data(file_path):
    """加载Excel数据文件"""
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    """数据预处理"""
    # 确保标签是整数类型
    df['label'] = df['label'].astype(int)
    
    # 将数据转换为HuggingFace数据集格式
    dataset = Dataset.from_pandas(df)
    return dataset

def prepare_model_and_tokenizer():
    """准备模型和分词器"""
    # model_name = "facebook/bart-large-mnli"
    model_name = '/your/path/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=4, problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # 将模型移动到指定设备
    device = get_device()
    model = model.to(device)
    logging.info(f"使用设备: {device}")
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """对文本进行分词"""
    return tokenizer(
        examples["paragraph"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    # 如果predictions是tuple或list，取第一个元素
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    predictions = np.array(predictions)
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

class CustomTrainer(Trainer):
    def log(self, logs):
        """重写日志记录方法"""
        if self.state.global_step % 100 == 0:  # 每100步记录一次
            logging.info(f"Step {self.state.global_step}: {json.dumps(logs)}")
        super().log(logs)

def main():
    # 加载数据
    data_path = "data/FileExtractorData.xlsx"
    logging.info(f"正在加载数据: {data_path}")
    df = load_data(data_path)
    
    # 数据预处理
    logging.info("正在进行数据预处理")
    df['label'] = df['label'].astype(int)
    
    # 划分训练集和验证集（在DataFrame上操作）
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )
    logging.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(eval_df)}")
    
    # 转换为HuggingFace数据集
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    
    # 准备模型和分词器
    logging.info("正在加载预训练模型和分词器")
    model, tokenizer = prepare_model_and_tokenizer()
    
    # 对数据集进行分词
    logging.info("正在对数据集进行分词")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # 增大batch size
        per_device_eval_batch_size=8,   # 增大batch size
        num_train_epochs=8,  # 增加训练轮数
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",  # 每个epoch保存一次
        eval_strategy="epoch",  # 每个epoch评估一次
        load_best_model_at_end=True,  # 加载最佳模型
        metric_for_best_model="accuracy",  # 使用accuracy作为评估指标
        greater_is_better=True,  # accuracy越大越好
        logging_steps=100,  # 每100步记录一次日志
        report_to="none",  # 不使用任何报告器
        gradient_accumulation_steps=2,  # 梯度累积
        dataloader_num_workers=4,  # 数据加载器的工作进程数
        dataloader_pin_memory=True  # 使用固定内存
    )
    
    # 创建数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 创建训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 开始训练
    logging.info("开始训练模型")
    trainer.train()
    
    # 保存模型
    model_save_path = "./models/final_model"
    logging.info(f"正在保存模型到: {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # 记录最终评估结果
    final_metrics = trainer.evaluate()
    logging.info(f"最终评估结果: {json.dumps(final_metrics)}")

if __name__ == "__main__":
    main()
