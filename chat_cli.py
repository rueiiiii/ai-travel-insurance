"""
命令列版聊天介面 - 用來驗證系統能用
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from dotenv import load_dotenv
load_dotenv()

from vector_store import VectorStoreManager
from rag_chain import RAGChain

print("🚀 載入 AI 旅平險專員...")

vector_mgr = VectorStoreManager(
    embedding_type=os.getenv("EMBEDDING_TYPE", "huggingface")
)
vector_mgr.load_index()

rag = RAGChain(
    vector_mgr,
    llm_type=os.getenv("LLM_TYPE", "gemini"),
    top_k=5,
)

print("\n" + "=" * 60)
print("✅ AI 旅平險專員已就緒！輸入 'quit' 結束")
print("=" * 60)

while True:
    question = input("\n❓ 你的問題：")
    if question.strip().lower() in ["quit", "exit", "q"]:
        print("👋 再見！")
        break
    if not question.strip():
        continue

    print("\n🤔 思考中...\n")
    result = rag.answer(question)

    print("💬 AI 回答：")
    print("-" * 60)
    print(result["answer"])
    print("\n 參考來源：")
    for i, src in enumerate(result["sources"][:3], 1):
        print(f"  {i}. {src['source']} 第 {src['page']} 頁 "
              f"(相似度: {src['similarity_score']:.3f})")