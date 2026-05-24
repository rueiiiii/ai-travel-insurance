# AI 旅平險專員

一套以 RAG (Retrieval-Augmented Generation)技術建立的旅遊平安險諮詢系統，  
讓使用者在投保前能快速了解保單內容，並附上明確條款引用來源。

---

## 系統特色

- **可溯源**：每個回答都附「條款來源 + 頁碼 + 相似度分數」
- **反幻覺**：找不到答案會誠實回應，不會編造
- **多保單支援**：可同時索引多家保險公司合約做交叉比對
- **免費部署**：使用 HuggingFace Embedding + Gemini 免費 API

---

## 系統架構

```
PDF 合約 → 文件解析 → 文本切塊 → 向量化 → ChromaDB
                                              ↓
使用者問題 → Query 向量化 → 相似度檢索 → Top-5 片段 → Gemini LLM
                                                          ↓
                                                附引用來源的回答
```

---

## 核心套件


| 類別 | 套件 |
|------|------|
| PDF 解析 | pdfplumber |
| RAG 框架 | LangChain |
| Embedding | HuggingFace paraphrase-multilingual-MiniLM-L12-v2 |
| 向量資料庫 | ChromaDB |
| LLM | Google Gemini 2.5 Flash |
| 開發環境 | Python 3.11 / VS Code |

---

## 實作

### 1. 安裝套件
```bash
pip install -r requirements.txt
```

### 2. 設定 API Key
複製 `.env.example` 為 `.env`，填入你的 Gemini API Key：
```
GOOGLE_API_KEY=你的金鑰
EMBEDDING_TYPE=huggingface
LLM_TYPE=gemini
```

### 3. 放入旅平險 PDF
將保險合約 PDF 放進 `data/` 資料夾。

### 4. 建立向量資料庫
```bash
python main.py build
```

### 5. 啟動聊天介面
```bash
python chat_cli.py
```

---

##  專案結構

```
ai_travel_insurance/
├── requirements.txt
├── .env.example         # 設定 API Key
├── main.py              # 主入口
├── chat_cli.py          # 命令列聊天介面
│
├── data/                # 保險 PDF (請自行放入)
├── chroma_db/           # 向量資料庫 (執行後產生)
│
├── src/
│   ├── document_processor.py   # PDF 解析 + 切塊
│   ├── vector_store.py         # 向量資料庫管理
│   ├── rag_chain.py            # RAG 問答鏈                  
│
└── evaluation/
    └── evaluate.py             # 評估腳本
```

---

##  驗證方法

本系統與 **Google NotebookLM** 進行對照比較，使用相同 PDF + 相同題目，量化評估：

- 條款引用率
- 答案精確度
- 跨保單推理能力

---

##  須知

- Gemini API 免費版有額度限制：每分鐘 10 次請求
- 首次執行會下載 HuggingFace Embedding 模型（約 500MB）
- 已測試環境：Python 3.11
---
