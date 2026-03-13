"""
测试智谱 AI 的 API 密钥是否正确。

使用方法：
1) 在项目根目录创建 .env 文件，并写入：
   ZHIPUAI_API_KEY=你的密钥
2) 运行：
   python test_zhipu_key.py
"""

from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv

load_dotenv()

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[{"role": "user", "content": "你好"}],
        stream=False,
    )
    print("✅ API 密钥验证成功！")
    print("回复:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ 密钥验证失败: {e}")

