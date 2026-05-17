"""
Gradio Web UI - AI 旅平險專員
=============================
提供使用者友善的問答介面
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from document_processor import InsurancePDFProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain

load_dotenv()

# ============================================
# 初始化系統 (只在啟動時跑一次)
# ============================================
print("🚀 啟動 AI 旅平險專員...")

EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "huggingface")  # 預設用免費版
LLM_TYPE = os.getenv("LLM_TYPE", "openai")

vector_mgr = VectorStoreManager(embedding_type=EMBEDDING_TYPE)

# 如果向量庫不存在,先建立
if not os.path.exists("./chroma_db"):
    print("⚠️  向量資料庫不存在,正在建立...")
    processor = InsurancePDFProcessor()
    docs = processor.process_directory("./data")
    vector_mgr.build_index(docs)
else:
    vector_mgr.load_index()

rag = RAGChain(vector_mgr, llm_type=LLM_TYPE, top_k=5)
print("✅ 系統就緒")


# ============================================
# 回答函式
# ============================================
def answer_question(question, chat_history):
    """處理使用者提問"""
    if not question.strip():
        return "", chat_history

    result = rag.answer(question)

    # 組合回答 + 來源
    answer = result["answer"]
    sources_md = "\n\n---\n**📚 引用來源:**\n"
    for i, src in enumerate(result["sources"][:3], 1):
        sources_md += (
            f"{i}. `{src['source']}` 第 {src['page']} 頁 "
            f"(相似度: {src['similarity_score']:.3f})\n"
        )

    full_answer = answer + sources_md
    chat_history.append((question, full_answer))
    return "", chat_history


# ============================================
# Gradio 介面
# ============================================
with gr.Blocks(
    title="AI 旅平險專員",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
        # 🛫 AI 旅平險專員
        ### 您專屬的旅遊平安險諮詢助手
        
        歡迎詢問任何關於旅平險的疑問,例如:
        - 行李遺失能賠多少?
        - 飛機延誤的理賠條件?
        - 國外就醫費用如何申請?
        - 哪些情況屬於除外責任?
        """
    )

    chatbot = gr.Chatbot(
        label="諮詢對話",
        height=500,
        show_copy_button=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="輸入您的問題",
            placeholder="例如:出國行李遺失最多可以賠多少?",
            scale=8,
        )
        submit_btn = gr.Button("送出", variant="primary", scale=1)
        clear_btn = gr.Button("清除", scale=1)

    gr.Examples(
        examples=[
            "出國時行李被偷,可以申請多少理賠?",
            "去程飛機延誤 5 小時,理賠金額多少?",
            "在國外發生車禍住院,可以理賠哪些項目?",
            "如果在巴黎遺失護照,保險會理賠嗎?",
            "戰爭、暴動造成的損失會理賠嗎?",
        ],
        inputs=msg,
        label="🔥 熱門問題範例",
    )

    msg.submit(answer_question, [msg, chatbot], [msg, chatbot])
    submit_btn.click(answer_question, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)

    gr.Markdown(
        """
        ---
        ⚠️ **免責聲明:** 此系統僅供參考,實際理賠仍應以保險公司條款為準。
        
        🛠️ 技術:RAG (Retrieval-Augmented Generation) | LangChain + ChromaDB + LLM
        """
    )


if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)