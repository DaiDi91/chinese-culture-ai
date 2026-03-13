"""
用于排查本地 / 网络加载中文向量模型的问题。

运行方式：
    python test_fix.py
"""

# 测试1：检查证书
import ssl

print(f"SSL版本: {ssl.OPENSSL_VERSION}")

# 测试2：尝试用镜像下载
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from sentence_transformers import SentenceTransformer

    # 先试本地
    try:
        model = SentenceTransformer("F:/models/bge-small-zh-v1.5")
        print("✅ 本地模型加载成功")
    except Exception:
        # 再试网络
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        print("✅ 网络下载成功")
except Exception as e:
    print(f"❌ 失败: {e}")
    print("建议: 1. 用本地模型路径 2. 用智谱API")

