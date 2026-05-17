"""
向量資料庫模組
功能：將文本片段向量化並存入 ChromaDB，提供相似度檢索
"""

import os
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# 備用：使用免費的 HuggingFace 模型
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """向量資料庫管理員"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_type: str = "openai",  # "openai" or "huggingface"
        collection_name: str = "travel_insurance",
    ):
        """
        Args:
            persist_directory: 向量資料庫儲存路徑
            embedding_type: 使用 OpenAI 或免費的 HuggingFace
            collection_name: collection 名稱
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 選擇 Embedding 模型
        if embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print("🔧 使用 OpenAI Embedding (text-embedding-3-small)")
        else:
            # 多語言模型，支援中文且免費
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("🔧 使用 HuggingFace 多語言 Embedding (免費)")

        self.vectorstore = None

    def build_index(self, documents: List[Document]):
        """建立向量索引"""
        print(f"🔨 開始建立向量索引 ({len(documents)} 個片段)...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        print(f"✅ 向量索引建立完成，儲存於: {self.persist_directory}")
        return self.vectorstore

    def load_index(self):
        """載入既有的向量索引"""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"向量資料庫不存在: {self.persist_directory}，請先呼叫 build_index()"
            )
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        print(f"📂 已載入向量索引: {self.persist_directory}")
        return self.vectorstore

    def search(self, query: str, k: int = 5) -> List[Document]:
        """搜尋相似的文本片段"""
        if self.vectorstore is None:
            self.load_index()
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # 將相似度分數加入 metadata
        docs_with_scores = []
        for doc, score in results:
            doc.metadata["similarity_score"] = float(score)
            docs_with_scores.append(doc)
        return docs_with_scores

    def get_retriever(self, k: int = 5):
        """取得 LangChain Retriever 介面"""
        if self.vectorstore is None:
            self.load_index()
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    from document_processor import InsurancePDFProcessor

    # 範例：建立完整流程
    processor = InsurancePDFProcessor()
    docs = processor.process_directory("./data")

    vector_mgr = VectorStoreManager(embedding_type="huggingface")
    vector_mgr.build_index(docs)

    # 測試查詢
    test_query = "出國時行李遺失可以賠多少錢?"
    results = vector_mgr.search(test_query, k=3)
    print(f"\n🔍 查詢: {test_query}")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 結果 {i} (相似度: {doc.metadata['similarity_score']:.4f}) ---")
        print(f"來源: {doc.metadata['source']} 第 {doc.metadata['page']} 頁")
        print(f"內容: {doc.page_content[:200]}")
