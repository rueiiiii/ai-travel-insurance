"""
主執行腳本
==========
一鍵跑完整個流程:資料處理 -> 向量化 -> 啟動 UI
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from dotenv import load_dotenv
load_dotenv()


def build_database():
    """步驟 1+2:處理 PDF 並建立向量資料庫"""
    from document_processor import InsurancePDFProcessor
    from vector_store import VectorStoreManager

    print("\n" + "=" * 60)
    print("📚 Step 1: 處理 PDF 文件")
    print("=" * 60)
    processor = InsurancePDFProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.process_directory("./data")

    if not documents:
        print("⚠️ 找不到任何 PDF,請將保險合約 PDF 放入 ./data/ 資料夾")
        return

    print("\n" + "=" * 60)
    print("🔮 Step 2: 建立向量資料庫")
    print("=" * 60)
    embedding_type = os.getenv("EMBEDDING_TYPE", "huggingface")
    vector_mgr = VectorStoreManager(embedding_type=embedding_type)
    vector_mgr.build_index(documents)
    print("\n✅ 資料庫建置完成!")


def run_app():
    """步驟 3:啟動 Web 介面"""
    print("\n" + "=" * 60)
    print("🚀 Step 3: 啟動 AI 旅平險專員 Web 介面")
    print("=" * 60)
    from app import demo
    demo.launch(share=True, inbrowser=True)


def run_evaluation():
    """步驟 4:評估與比較分析"""
    print("\n" + "=" * 60)
    print("🧪 Step 4: 執行評估分析")
    print("=" * 60)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/evaluation")
    from evaluate import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate_our_system()
    evaluator.load_notebooklm_answers()
    evaluator.generate_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 旅平險專員")
    parser.add_argument(
        "command",
        choices=["build", "run", "eval", "all"],
        help="build=建資料庫 / run=啟動UI / eval=跑評估 / all=全部",
    )
    args = parser.parse_args()

    if args.command == "build":
        build_database()
    elif args.command == "run":
        run_app()
    elif args.command == "eval":
        run_evaluation()
    elif args.command == "all":
        build_database()
        run_app()
