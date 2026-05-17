import os
import re
from pathlib import Path
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class InsurancePDFProcessor:
    """旅平險合約 PDF 處理器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 每個切塊的字數
            chunk_overlap: 切塊之間重疊的字數
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 中文友善的分隔符號：條款、項、款
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n第", "\n第",  # 條款分割
                "\n\n", "\n",
                "。", "；", "，",
                " ", ""
            ],
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        從 PDF 萃取文字，並保留頁碼資訊
        
        Returns:
            [{"page": 1, "text": "...", "source": "filename.pdf"}, ...]
        """
        results = []
        pdf_name = Path(pdf_path).name

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # 同時抓取表格 (理賠金額表常以表格形式呈現)
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        row_text = " | ".join(
                            [str(cell) if cell else "" for cell in row]
                        )
                        table_text += row_text + "\n"

                full_text = text + "\n" + table_text if table_text else text
                cleaned_text = self.clean_text(full_text)

                if cleaned_text.strip():
                    results.append({
                        "page": page_num,
                        "text": cleaned_text,
                        "source": pdf_name,
                    })
        return results

    def clean_text(self, text: str) -> str:
        """清理多餘空白、頁碼、頁眉頁腳"""
        # 移除多餘空白
        text = re.sub(r"\s+", " ", text)
        # 移除頁碼類字串
        text = re.sub(r"第\s*\d+\s*頁\s*[/／]\s*共\s*\d+\s*頁", "", text)
        # 移除單獨的數字頁碼 (出現在行首/行末的單獨數字)
        text = re.sub(r"\s+\d{1,3}\s+", " ", text)
        return text.strip()

    def chunk_documents(self, pages: List[Dict]) -> List[Document]:
        """
        將解析好的頁面切塊，產出 LangChain Document 物件
        
        每個 chunk 都帶有 metadata: source, page, chunk_id
        - 方便最後回答時能引用「依據第 X 條 第 Y 項」
        """
        all_chunks = []
        chunk_id = 0

        for page_data in pages:
            page_text = page_data["text"]
            # 切塊
            chunks = self.text_splitter.split_text(page_text)
            for chunk_text in chunks:
                if len(chunk_text.strip()) < 20:  # 過濾過短的片段
                    continue
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": page_data["source"],
                        "page": page_data["page"],
                        "chunk_id": chunk_id,
                    },
                )
                all_chunks.append(doc)
                chunk_id += 1
        return all_chunks

    def process_directory(self, pdf_dir: str) -> List[Document]:
        """處理整個目錄中的所有 PDF"""
        pdf_dir = Path(pdf_dir)
        all_documents = []

        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f" 找到 {len(pdf_files)} 份保險合約 PDF")

        for pdf_path in pdf_files:
            print(f"    處理中: {pdf_path.name}")
            pages = self.extract_text_from_pdf(str(pdf_path))
            chunks = self.chunk_documents(pages)
            all_documents.extend(chunks)
            print(f"     ✓ 切成 {len(chunks)} 個片段")

        print(f" 總共產生 {len(all_documents)} 個文本片段")
        return all_documents


if __name__ == "__main__":
    # 測試
    processor = InsurancePDFProcessor(chunk_size=500, chunk_overlap=50)
    docs = processor.process_directory("./data")
    if docs:
        print("\n 第一個 chunk 範例:")
        print(f"   來源: {docs[0].metadata}")
        print(f"   內容: {docs[0].page_content[:200]}...")
