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
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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
    model_name = "facebook/bart-large-mnli"
    # model_name = '/your/path/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce'
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
    # 加载 HuggingFace 数据集
    dataset_name = os.getenv("DATASET_NAME")
    if not dataset_name:
        raise ValueError("请在环境变量中设置 DATASET_NAME")
    logging.info(f"正在加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    logging.info(f"数据集结构: {dataset}")

    # 取出各个 split
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}, 测试集大小: {len(test_dataset)}")

    # 数据预处理（确保 label 为 int）
    def cast_label(example):
        example["label"] = int(example["label"])
        return example
    train_dataset = train_dataset.map(cast_label)
    eval_dataset = eval_dataset.map(cast_label)
    test_dataset = test_dataset.map(cast_label)

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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
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
