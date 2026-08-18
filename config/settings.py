"""
项目统一配置文件
所有模块都应该从这里导入配置，而不是使用子模块的配置文件
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据库目录（SQLite 数据库文件存储位置）
DB_DIR = PROJECT_ROOT / "db"
DB_DIR.mkdir(exist_ok=True)  # 自动创建目录

# ==================== API 回调配置 ====================
LOG_CALLBACK_API_URL = 'http://172.16.99.67:8088/sw/app/progressInfoForCustomizedGenApp'
TR_GEN_URL = 'http://172.16.99.67:8088/sw/idep/plugin/logCallback'
FINISH_URL = 'http://172.16.99.67:8088/sw/idep/plugin/exeFinishCallback'
CALL_GEN_REQ_REQUEST = "http://172.16.99.67:8378/tc/tools/parseAndSaveTr"

# ==================== LLM 模型配置 ====================
# 用于 doc_service 模块的大语言模型配置
LLM_MODEL_PROVIDER = "openai"
LLM_MODEL = "qwen-plus"
LLM_API_KEY = "sk-a5ad92221a5945e2952bbd23dfffe2a0"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TEMPERATURE = 0

# ==================== Embedding 模型配置 ====================
# 用于文档嵌入的向量模型配置
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_API_KEY = "sk-a5ad92221a5945e2952bbd23dfffe2a0"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"



# 用于 LangChain Milvus 向量存储的配置
MILVUS_VECTOR_STORE_CONFIG = {
    "uri": "http://127.0.0.1:19530",  # Milvus 服务地址
    "db_name": "default",  # 数据库名称
    "collection_name": "default",  # 集合名称（默认，可通过参数覆盖）
    "index_type": "FLAT",  # 索引类型：FLAT（精确搜索，适合小数据集）或 IVF_FLAT（适合大数据集）
    "metric_type": "L2",  # 距离度量：L2（欧氏距离）或 IP（内积），text-embedding-v4 通常用 L2
    "drop_old": True  # 是否覆盖模式：如果集合已存在，先删除旧集合
}

# ==================== API 服务配置 ====================
API_CONFIG = {
    "upload_folder": "/tmp/uploads",
    "supported_formats": [".txt", ".pdf", ".md", ".docx"],
    "max_content_length": 16 * 1024 * 1024,  # 16MB
}