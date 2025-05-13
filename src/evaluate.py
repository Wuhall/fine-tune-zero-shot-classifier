import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

class ZeroShotClassifier:
    def __init__(self, model_path):
        """初始化分类器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射
        self.label_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3
        }
    
    def predict(self, text):
        """对单个文本进行预测"""
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        # 转换标签
        predicted_label = self.label_map[predicted_class]
        confidence = predictions[0][predicted_class].item()
        
        return {
            "label": predicted_label,
            "confidence": confidence
        }
    
    def predict_batch(self, texts):
        """对多个文本进行批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    # 加载数据
    data_path = "data/FileExtractorData.xlsx"
    df = pd.read_excel(data_path)

    # 加载模型
    model_path = "./models/checkpoints/checkpoint-34"
    classifier = ZeroShotClassifier(model_path)

    # 预测并对比
    print(f"共{len(df)}条数据，开始推理并对比：")
    error_count = 0
    true_positive_count = 0
    total_positive = 0
    for idx, row in df.iterrows():
        text = row['paragraph']
        true_label = row['label']
        pred_result = classifier.predict(text)
        pred_label = pred_result['label']
        confidence = pred_result['confidence']
        if pred_label != true_label:
            error_count += 1
            print(f"样本{idx+1}：")
            print(f"文本: {text}")
            print(f"真实标签: {true_label}，预测标签: {pred_label}，置信度: {confidence:.4f}")
            print(f"{'正确' if true_label == pred_label else '错误'}")
            print("-" * 60)
        if true_label < 3:
            total_positive += 1
        if pred_label < 3 and true_label < 3:
            true_positive_count += 1
    print(f"分类正确率: {(len(df) - error_count) / len(df)}")
    print(f"总正样本数: {total_positive}")
    print(f"正确提取率: {true_positive_count / total_positive}")   
if __name__ == "__main__":
    main() 