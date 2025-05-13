import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# FastAPI部分
app = FastAPI(
    title="零样本文本分类服务",
    version="1.0.0",
    docs_url="/docs"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: int
    confidence: float

# 加载模型（只加载一次）
classifier = ZeroShotClassifier("models/checkpoints/checkpoint-34")

@app.post("/predict", response_model=PredictResponse, summary="文本分类预测")
def predict_api(request: PredictRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="请提供text字段")
    result = classifier.predict(request.text)
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False) 