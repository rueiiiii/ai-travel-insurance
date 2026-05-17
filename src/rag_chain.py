"""
RAG 問答鏈
==========
功能：整合檢索 + LLM，產生有引用來源的回答
"""

import os
from typing import List, Dict
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


# ============================================
# 旅平險 AI 專員 - Prompt 設計 (核心)
# ============================================
INSURANCE_AGENT_PROMPT = """你是一位專業的「旅平險 AI 保險專員」，協助使用者了解保險合約內容。

請嚴格遵守以下規則：
1. **僅根據下方提供的「保險合約片段」回答**，不要編造合約上沒有的內容
2. 回答時必須**標註引用來源** (例如:「依據第 12 條」或「(來源: XXX保險.pdf 第3頁)」)
3. 若合約中有相關但條件不同的情況 (例如「出發延誤」vs「回程延誤」),要詳細分辨
4. 若涉及金額或日數,請完整列出條件 (例如:「住院滿 3 天起算,每日 1000 元,最高 30 天」)
5. 若合約片段中找不到答案,直接回答:「依目前提供的條款,無法找到明確規範,建議直接洽詢保險公司確認」
6. 回答以繁體中文回應,語氣專業但親切

==============================
保險合約片段:
{context}
==============================

使用者問題: {question}

請依據上述合約片段給出專業回答:"""


class RAGChain:
    """RAG 問答鏈"""

    def __init__(
        self,
        vector_store_manager,
        llm_type: str = "openai",  # "openai" or "gemini"
        model_name: str = None,
        top_k: int = 5,
        temperature: float = 0.1,
    ):
        """
        Args:
            vector_store_manager: VectorStoreManager 實例
            llm_type: 使用的 LLM 廠商
            model_name: 模型名稱
            top_k: 檢索回傳的片段數
            temperature: 創造力 (0=最嚴謹)
        """
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k

        # 選擇 LLM
        if llm_type == "openai":
            self.llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print(f"🤖 使用 OpenAI {model_name or 'gpt-4o-mini'}")
        elif llm_type == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-flash",
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            print(f"🤖 使用 Google {model_name or 'gemini-2.5-flash'}")

        # Prompt 模板
        self.prompt = ChatPromptTemplate.from_template(INSURANCE_AGENT_PROMPT)

    def format_docs(self, docs: List[Document]) -> str:
        """格式化檢索到的文件,加上來源標記"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("similarity_score", 0)
            formatted.append(
                f"[片段 {i}] (來源: {source} 第 {page} 頁, 相似度: {score:.3f})\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(formatted)

    def answer(self, question: str) -> Dict:
        """
        回答問題,回傳 dict 包含:
        - answer: AI 回答
        - sources: 引用的來源片段
        - question: 原始問題
        """
        # 1. 檢索相關片段
        retrieved_docs = self.vector_store_manager.search(question, k=self.top_k)

        # 2. 組合 Prompt
        context = self.format_docs(retrieved_docs)
        chain = self.prompt | self.llm | StrOutputParser()

        # 3. 取得 LLM 回答
        answer_text = chain.invoke({
            "context": context,
            "question": question,
        })

        # 4. 回傳完整資訊
        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "similarity_score": doc.metadata.get("similarity_score"),
                    "content": doc.page_content[:200] + "...",
                }
                for doc in retrieved_docs
            ],
        }


if __name__ == "__main__":
    from document_processor import InsurancePDFProcessor
    from vector_store import VectorStoreManager
    from dotenv import load_dotenv

    load_dotenv()

    # 載入既有的向量庫
    vector_mgr = VectorStoreManager(embedding_type="openai")
    vector_mgr.load_index()

    # 建立 RAG
    rag = RAGChain(vector_mgr, llm_type="openai")

    # 測試問題
    questions = [
        "出國時行李遺失最多可以賠多少錢?",
        "如果回程飛機延誤超過 4 小時,賠償內容是什麼?",
        "在國外住院超過 3 天,可以申請哪些理賠?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ 問題: {q}")
        print(f"{'='*60}")
        result = rag.answer(q)
        print(f"\n💬 回答:\n{result['answer']}")
        print(f"\n📚 參考來源:")
        for src in result["sources"][:3]:
            print(f"  - {src['source']} 第 {src['page']} 頁 (相似度: {src['similarity_score']:.3f})")
