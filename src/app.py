import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from typing import List, Dict, Any

import numpy as np
import faiss  # pyright: ignore[reportMissingImports]
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from zhipuai import ZhipuAI
import gradio as gr


# =========================
# 配置区
# =========================

# 项目根目录（根据当前文件位置自动推断）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 知识库文本文件路径
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "texts", "knowledge_base.txt")

# 图片所在目录（假定图片文件与知识库描述中的图片名对应）
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

# FAISS 向量库持久化目录（可选，当前实现每次启动从知识库重建）
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")


# =========================
# FAISS 向量存储
# =========================


class SimpleVectorStore:
    """
    基于 FAISS 的简单向量存储，支持按向量检索并返回对应文本与元数据。
    """

    def __init__(self, dimension: int):
        """
        dimension: 向量维度（需与嵌入模型输出一致，如智谱 embedding-2）
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """添加文本及其向量与元数据。"""
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts、embeddings、metadatas 长度必须一致")
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """
        检索与 query_embedding 最接近的 k 条记录。
        返回列表，每项为 {"title", "image", "text", "score"}，score 为 L2 距离。
        """
        if not self.documents:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(q, min(k, len(self.documents)))
        hits: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            meta = self.metadatas[idx]
            hits.append({
                "title": meta.get("title", ""),
                "image": meta.get("image", ""),
                "text": self.documents[idx],
                "score": float(distances[0][i]),
            })
        return hits


# =========================
# 工具函数
# =========================

def load_env() -> str:
    """
    从 .env 文件加载环境变量，并返回智谱 API Key。

    支持 .env 中任一种写法：
    ZHIPU_API_KEY=你的密钥
    或
    ZHIPUAI_API_KEY=你的密钥

    若出现 401「令牌已过期或验证不正确」，请到智谱开放平台重新生成密钥并更新 .env。
    """
    load_dotenv()
    api_key = (
        os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "未在 .env 或环境变量中找到 ZHIPU_API_KEY / ZHIPUAI_API_KEY，请先配置。"
        )
    return api_key


def parse_knowledge_base(file_path: str) -> List[Dict[str, Any]]:
    """
    解析知识库文本文件。

    预期文件格式示例（多条内容顺序排列）：

    **中国茶文化**
    image: tea.jpg
    description: 中国茶文化历史悠久，包含绿茶、红茶等多种茶类……

    **京剧**
    image: jingju.jpg
    description: 京剧是中国传统戏曲艺术的代表之一……

    解析规则：
    - 以 "**标题**" 作为一条记录的开始
    - 紧随其后应该有 "image: xxx.jpg" 和 "description: 文本内容"
    - 中间允许有空行
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"知识库文件不存在：{file_path}")

    entries: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    def flush_current():
        """将当前记录加入列表（如果有效）"""
        nonlocal current
        if current.get("title") and current.get("description"):
            # full_text 用于向量化检索，包含标题和描述
            current["full_text"] = f"{current['title']}\n{current['description']}"
            entries.append(current)
        current = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                # 空行直接跳过
                continue

            # 标题行：以 "**" 开头和结尾
            if line.startswith("**") and line.endswith("**") and len(line) > 4:
                # 遇到新的标题前，先把上一条记录存入
                flush_current()
                title = line.strip("*").strip()
                current["title"] = title
                continue

            # 图片行：image: xxx.jpg
            if line.lower().startswith("image:"):
                image_name = line.split(":", 1)[1].strip()
                current["image"] = image_name
                continue

            # 描述行：description: 描述内容
            if line.lower().startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                # 如果文件中描述可能跨多行，这里可以按行累加
                if "description" in current:
                    current["description"] += "\n" + desc
                else:
                    current["description"] = desc
                continue

            # 其他行，如果属于描述的补充内容，则追加到 description
            if "description" in current:
                current["description"] += "\n" + line

    # 文件结束后，处理最后一条记录
    flush_current()
    return entries


def _get_embeddings(client: ZhipuAI, texts: List[str]) -> List[List[float]]:
    """调用智谱 embedding-2 对多条文本取向量，按需分批以符合 API 限制。"""
    if not texts:
        return []
    # 单次请求条数不宜过多，这里每批最多 25 条
    batch_size = 25
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(model="embedding-2", input=batch)
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "令牌" in err_msg or "Authentication" in type(e).__name__:
                raise RuntimeError(
                    "智谱 API 密钥无效或已过期（401）。请到 https://open.bigmodel.cn/ 重新生成 API Key，"
                    "并更新项目根目录 .env 中的 ZHIPU_API_KEY 或 ZHIPUAI_API_KEY。"
                ) from e
            raise
        all_embeddings.extend([item.embedding for item in resp.data])
    return all_embeddings


def build_or_load_vector_store(entries: List[Dict[str, Any]]):
    """
    使用 FAISS 构建向量存储，并用智谱 AI（embedding-2）生成向量。

    每次启动时从知识库重新解析并写入，保证与 data/texts/knowledge_base.txt 一致。
    """

    api_key = load_env()
    zhipu_client = ZhipuAI(api_key=api_key)

    texts = [e["full_text"] for e in entries]
    metadatas = [
        {"title": e.get("title", ""), "image": e.get("image", "")}
        for e in entries
    ]

    if not texts:
        # 无内容时仍返回一个空 store，维度用智谱 embedding-2 的常见维度
        vector_store = SimpleVectorStore(dimension=1024)
        return zhipu_client, vector_store

    embeddings = _get_embeddings(zhipu_client, texts)
    dimension = len(embeddings[0])
    vector_store = SimpleVectorStore(dimension=dimension)
    vector_store.add_texts(texts, embeddings, metadatas)

    return zhipu_client, vector_store


def search_knowledge(
    vector_store: SimpleVectorStore,
    zhipu_client: ZhipuAI,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    在 FAISS 向量库中检索与用户问题最相关的若干条知识。

    使用智谱 embedding-2 对 query 取向量后，再在 FAISS 中做最近邻检索。
    返回列表，每个元素为包含 title / image / text / score 的字典。
    """
    if not query.strip():
        return []
    query_embeddings = _get_embeddings(zhipu_client, [query])
    if not query_embeddings:
        return []
    return vector_store.search(query_embeddings[0], k=top_k)


def create_zhipu_client() -> ZhipuAI:
    """
    创建智谱 AI 客户端。
    """
    api_key = load_env()
    client = ZhipuAI(api_key=api_key)
    return client


def generate_answer_with_zhipu(
    client: ZhipuAI, question: str, contexts: List[Dict[str, Any]]
) -> str:
    """
    使用智谱 AI，根据检索到的知识上下文生成回答。

    参数：
    - client: ZhipuAI 客户端实例
    - question: 用户问题
    - contexts: 来自向量库的若干条相关知识，每条包含 title、text 等
    """
    # 将检索结果拼接为一个上下文字符串
    context_texts = []
    for c in contexts:
        piece = f"【标题】{c.get('title', '')}\n{c.get('text', '')}"
        context_texts.append(piece)
    context_str = "\n\n".join(context_texts)

    # 如果没有检索到知识，也直接把用户问题发给模型，
    # 并在系统提示中声明知识可能不全。
    system_prompt = (
        "你是一位精通中华文化的智能助手，包括历史、哲学、艺术、节日、习俗等。\n"
        "请优先基于提供的知识内容进行回答，如果知识中没有明确的信息，可以结合你已有的通识做适度扩展，"
        "但请保持严谨、友好，并使用简体中文回答。"
    )

    user_prompt = (
        "以下是与用户问题最相关的知识片段，请在回答时尽量引用其中的信息：\n\n"
        f"{context_str}\n\n"
        f"用户问题：{question}\n\n"
        "请给出一段清晰、有结构的回答，可以适当分点说明。"
    )

    resp = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # 不同 SDK 版本字段可能略有不同，这里按官方较新的字段名处理：
    try:
        answer = resp.choices[0].message.content
    except Exception:
        # 兜底：如果结构略有差异，尝试其它访问方式
        answer = str(resp)

    return answer


def _load_image_from_url(url: str) -> Image.Image:
    """
    从 URL 下载图片并转为 PIL.Image（便于 Gradio 直接显示）。
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def _load_image_from_path(path: str) -> Image.Image:
    """
    从本地路径读取图片并转为 PIL.Image。
    """
    return Image.open(path).convert("RGB")


# =========================
# Gradio Web 界面
# =========================

def build_gradio_interface():
    """
    构建 Gradio 网页界面：

    - 「知识问答」：左侧文字回答 + 右侧图片
      · 优先使用知识库里配置的图片
      · 如果该条知识没有图片，则自动调用 AI 作图生成插画
    - 「AI 作图」：单独的文生图工具
    """
    # 1. 加载知识库并构建 FAISS 向量库（使用智谱 embedding-2）
    entries = parse_knowledge_base(KNOWLEDGE_FILE)
    zhipu_client, vector_store = build_or_load_vector_store(entries)

    # ---------- AI 作图 ----------
    def ai_draw_pipeline(prompt: str):
        """
        使用智谱 AI 文生图模型（如 cogview-3）根据文字提示生成图片。

        注意：不同 SDK 版本的 images 接口可能略有差异，如果报错，
        请根据你本地的 zhipuai 文档调整调用参数。
        """
        if not prompt or not prompt.strip():
            return None

        try:
            # 按智谱新版 SDK 的文生图接口调用（cogview-3）
            # 参考实现通常为：client.images.generations(model="cogview-3", prompt="...")
            resp = zhipu_client.images.generations(
                model="cogview-3",
                prompt=prompt,
            )
        except Exception as e:
            # 返回 None，界面会显示空图片；同时在终端中打印错误信息
            print("生成图片失败：", e)
            return None

        try:
            # SDK 通常返回图片 URL，这里下载为 PIL.Image 便于 Gradio 直接显示
            image_url = resp.data[0].url
            return _load_image_from_url(image_url)
        except Exception:
            try:
                # 如果是 base64，可以让 Gradio 直接识别 base64 字符串
                b64 = resp.data[0].b64_json
                # 部分 Gradio 版本对 data-url 支持不一致，优先走 url 下载；这里兜底返回 None
                return None
            except Exception:
                return None

    # ---------- 知识问答（结合知识检索 + AI 搜索 + AI 作图） ----------
    def qa_pipeline(user_question: str):
        """
        知识问答主流程：
        1）先用向量检索本地知识库
        2）结合检索结果 + 大模型进行回答（相当于「AI 搜索」增强版）
        3）图片优先使用知识库配置；若没有，则用 AI 作图生成插画
        """
        if not user_question or not user_question.strip():
            return "请先输入你想了解的中华文化问题。", None

        # 1）FAISS 向量检索（知识库语义搜索）
        hits = search_knowledge(vector_store, zhipu_client, user_question, top_k=3)

        # 2）调用智谱 AI 生成回答（相当于知识问答 + AI 搜索）
        answer = generate_answer_with_zhipu(zhipu_client, user_question, hits)

        # 3）图片逻辑：
        #    - 如果最相关知识条目有配置 image，则优先使用本地图片
        #    - 否则尝试调用 AI 作图，根据用户问题生成一张相关插画
        image_obj = None
        used_kb_image = False

        if hits:
            best = hits[0]
            image_name = best.get("image")
            if image_name:
                candidate_path = os.path.join(IMAGE_DIR, image_name)
                if os.path.exists(candidate_path):
                    image_obj = _load_image_from_path(candidate_path)
                    used_kb_image = True

        # 如果知识库没有合适图片，则用 AI 作图补充
        if not used_kb_image:
            # 提示词可根据需要更细化，这里直接用用户问题
            image_obj = ai_draw_pipeline(
                f"为下面的中华文化内容生成一张风格统一的插画：{user_question}"
            )

        return answer, image_obj

    css = """
    /* 强制暗黑风格（尽量兼容旧版 Gradio） */
    :root { color-scheme: dark; }
    body, .gradio-container { background: #0b1220 !important; color: #e5e7eb !important; }
    textarea, input, .wrap, .gr-box, .block, .panel {
        background: rgba(15, 23, 42, 0.75) !important;
        border-color: rgba(148,163,184,0.22) !important;
    }
    label, .prose, .prose * { color: #e5e7eb !important; }
    .cc-container { max-width: 1200px; margin: 0 auto; }
    .cc-hero {
        padding: 18px 18px 12px 18px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(245,158,11,0.12));
        border: 1px solid rgba(148,163,184,0.25);
    }
    .cc-title { font-size: 24px; font-weight: 800; line-height: 1.2; margin: 0; }
    .cc-sub { margin-top: 8px; color: rgba(203,213,225,0.9); }
    .cc-kbd {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.35);
        background: rgba(15, 23, 42, 0.55);
        font-size: 12px;
        margin-right: 6px;
    }
    .cc-card {
        padding: 14px;
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.25);
        background: rgba(2, 6, 23, 0.35);
        backdrop-filter: blur(6px);
    }
    .cc-btn-primary button {
        background: linear-gradient(135deg, #dc2626, #f59e0b) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
    }
    .cc-btn-primary button:hover { filter: brightness(1.02); }
    """

    with gr.Blocks(
        title="中华文化智能问答",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="amber",
            neutral_hue="slate",
            radius_size="lg",
            font=["ui-sans-serif", "system-ui", "Microsoft YaHei", "PingFang SC", "Segoe UI"],
        ),
        css=css,
    ) as demo:
        with gr.Column(elem_classes=["cc-container"]):
            gr.HTML(
                """
                <div class="cc-hero">
                  <div class="cc-title">中华文化智能问答系统</div>
                  <div class="cc-sub">
                    <span class="cc-kbd">知识检索</span>
                    <span class="cc-kbd">AI 回答</span>
                    <span class="cc-kbd">缺图自动作画</span>
                    <span class="cc-kbd">CogView 作图</span>
                    <div style="margin-top:8px;">
                      提示：在「知识问答」里提问会优先检索知识库；若该条目没有配图，会自动生成一张相关插画。
                    </div>
                  </div>
                </div>
                """
            )

        with gr.Tab("知识问答"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["cc-card"]):
                    question = gr.Textbox(
                        label="请输入你的问题",
                        placeholder="例如：端午节有哪些传统习俗？",
                        lines=3,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("提交问题", elem_classes=["cc-btn-primary"])
                        clear_btn = gr.Button("清空")
                    answer_box = gr.Textbox(
                        label="AI 回答",
                        lines=12,
                        interactive=False,
                    )
                    gr.Examples(
                        examples=[
                            "端午节有哪些传统习俗？",
                            "中国茶文化的特点是什么？",
                            "京剧的行当有哪些？分别代表什么？",
                        ],
                        inputs=[question],
                        label="示例问题（点击即可填入）",
                    )
                with gr.Column(scale=2, elem_classes=["cc-card"]):
                    image_box = gr.Image(
                        label="相关图片 / 自动生成插画",
                        type="pil",
                        height=420,
                    )

            submit_btn.click(
                fn=qa_pipeline,
                inputs=[question],
                outputs=[answer_box, image_box],
            )
            clear_btn.click(lambda: ("", "", None), None, [question, answer_box, image_box])

            # 支持按 Enter 触发
            question.submit(
                fn=qa_pipeline,
                inputs=[question],
                outputs=[answer_box, image_box],
            )

        with gr.Tab("AI 作图"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["cc-card"]):
                    draw_prompt = gr.Textbox(
                        label="作图提示词",
                        placeholder="例如：一幅水墨风格的长城日出图",
                        lines=4,
                    )
                    with gr.Row():
                        draw_btn = gr.Button("生成图片", elem_classes=["cc-btn-primary"])
                        draw_clear_btn = gr.Button("清空")
                    gr.Markdown(
                        "小技巧：可以加上风格与元素，例如“水墨风、留白、宣纸质感、暖色晨光、细节丰富”。"
                    )
                    gr.Examples(
                        examples=[
                            "水墨风格的长城日出，宣纸质感，留白，细节丰富",
                            "古风插画：汉服少女在灯会中，红灯笼，暖色光，精致",
                            "中国龙盘旋在祥云之间，国风插画，庄严，高清细节",
                        ],
                        inputs=[draw_prompt],
                        label="示例提示词（点击即可填入）",
                    )
                with gr.Column(scale=2, elem_classes=["cc-card"]):
                    draw_image = gr.Image(label="生成结果", type="pil", height=420)

            draw_btn.click(
                fn=ai_draw_pipeline,
                inputs=[draw_prompt],
                outputs=[draw_image],
            )
            draw_clear_btn.click(lambda: ("", None), None, [draw_prompt, draw_image])

    return demo


def main():
    """
    主入口：启动 Gradio 服务。
    """
    demo = build_gradio_interface()
    # server_name 设置为 0.0.0.0 方便在局域网访问（如果只在本机使用，也可以省略）
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

