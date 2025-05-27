import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# 加载 .env 文件
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 读取 Excel
df = pd.read_excel("FileExtractorData3.xlsx")

# 划分数据集（80% train, 10% validation, 10% test）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# 转为 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# 组合为 DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# 创建数据集仓库
api = HfApi()
repo_id = os.getenv("DATASET_NAME")
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=hf_token)
except Exception as e:
    print("Repo may already exist:", e)

# 上传数据集到 Hugging Face
dataset_dict.push_to_hub(repo_id, private=True, token=hf_token)