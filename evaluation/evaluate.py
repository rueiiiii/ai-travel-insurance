"""
評估與分析模組 (重點!)
=====================
功能:
1. 設計測試題庫 (Ground Truth)
2. 跑自家 RAG 系統的答案
3. 與 NotebookLM 答案做比較分析
4. 產出評估報告

評分標準對應:你必須使用 NotebookLM 或其他前沿AI應用的結果,
和你自己的系統做比較,並進行分析。

作者:朱政安
"""

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src")

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from vector_store import VectorStoreManager
from rag_chain import RAGChain

load_dotenv()


# ============================================
# 測試題庫設計 (Ground Truth)
# ============================================
# 注意:這些題目的「正確答案」需要你親自閱讀 PDF 後填入
# 否則無法做正確性驗證!
TEST_CASES = [
    {
        "id": 1,
        "question": "出國時行李遺失可以理賠多少錢?",
        "ground_truth": "依保單條款,行李損失通常每件最高賠償約 NT$ 3,000-5,000,總額上限約 NT$ 30,000-60,000 (依保單方案而定)",
        "category": "理賠金額",
        "difficulty": "easy",
    },
    {
        "id": 2,
        "question": "去程飛機延誤 4 小時,跟回程延誤 4 小時,理賠有什麼不同?",
        "ground_truth": "(需從 PDF 中查找確切答案) 部分保單兩者賠償相同,部分保單僅理賠回程",
        "category": "語意歧義測試",
        "difficulty": "hard",
    },
    {
        "id": 3,
        "question": "如果在日本住院滿 3 天,可以申請哪些補助?",
        "ground_truth": "(需從 PDF 中查找) 通常包含海外醫療費用、住院日額補助,部分保單對特定國家有加成",
        "category": "多層條件推理",
        "difficulty": "hard",
    },
    {
        "id": 4,
        "question": "戰爭或暴動造成的損失會理賠嗎?",
        "ground_truth": "通常列為除外責任,不予理賠",
        "category": "除外責任",
        "difficulty": "medium",
    },
    {
        "id": 5,
        "question": "保險的保障期間從什麼時候開始?",
        "ground_truth": "通常從離開居住地、或飛機起飛時起算",
        "category": "基本條款",
        "difficulty": "easy",
    },
    {
        "id": 6,
        "question": "如果我在國外發生意外身故,受益人是誰?",
        "ground_truth": "依保單上指定的受益人,未指定則為法定繼承人",
        "category": "受益人條款",
        "difficulty": "medium",
    },
    {
        "id": 7,
        "question": "懷孕期間出國旅遊發生意外有理賠嗎?",
        "ground_truth": "(需查) 通常懷孕、生產相關醫療列為除外責任",
        "category": "除外責任",
        "difficulty": "medium",
    },
    {
        "id": 8,
        "question": "從事高山攀岩活動受傷,可以理賠嗎?",
        "ground_truth": "(需查) 通常極限運動屬除外責任,需額外加保",
        "category": "除外責任",
        "difficulty": "medium",
    },
    {
        "id": 9,
        "question": "信用卡買機票送的旅平險,可以再買其他保單嗎?",
        "ground_truth": "(需查) 通常可以,且醫療險為實支實付,可累計;意外身故險各保單合計給付",
        "category": "保單疊加",
        "difficulty": "hard",
    },
    {
        "id": 10,
        "question": "出國前已經有的慢性病,旅平險會理賠嗎?",
        "ground_truth": "通常既往症列為除外責任,但部分保單有部分理賠",
        "category": "除外責任",
        "difficulty": "medium",
    },
]


# ============================================
# 評估指標
# ============================================
def keyword_coverage(answer: str, ground_truth: str) -> float:
    """關鍵字覆蓋率:檢查回答中是否包含 ground truth 的關鍵詞"""
    import re
    # 從 ground truth 抽取關鍵詞 (中文+數字)
    keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}|\d+", ground_truth))
    if not keywords:
        return 0.0
    hit = sum(1 for kw in keywords if kw in answer)
    return hit / len(keywords)


def has_citation(answer: str) -> bool:
    """檢查回答中是否含有條款引用"""
    import re
    # 偵測常見的引用形式
    patterns = [
        r"第\s*\d+\s*條",      # 第 X 條
        r"第\s*\d+\s*項",      # 第 X 項
        r"第\s*\d+\s*款",      # 第 X 款
        r"來源.*?頁",           # (來源: ... 頁)
        r"依.*?條款",           # 依...條款
        r"條款第",
    ]
    return any(re.search(p, answer) for p in patterns)


def answer_length(answer: str) -> int:
    """回答長度 (字數)"""
    return len(answer)


# ============================================
# 評估主流程
# ============================================
class Evaluator:
    """評估器"""

    def __init__(self):
        self.results = []

    def evaluate_our_system(self, test_cases=None):
        """執行自家 RAG 系統"""
        if test_cases is None:
            test_cases = TEST_CASES

        print("=" * 60)
        print("🧪 開始評估「我們的 RAG 系統」")
        print("=" * 60)

        vector_mgr = VectorStoreManager(
            embedding_type=os.getenv("EMBEDDING_TYPE", "huggingface")
        )
        vector_mgr.load_index()
        rag = RAGChain(vector_mgr, llm_type=os.getenv("LLM_TYPE", "openai"), top_k=5)

        for case in test_cases:
            print(f"\n[{case['id']}] {case['question']}")
            result = rag.answer(case["question"])
            answer = result["answer"]

            evaluation = {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "ground_truth": case["ground_truth"],
                "our_answer": answer,
                "our_keyword_coverage": keyword_coverage(answer, case["ground_truth"]),
                "our_has_citation": has_citation(answer),
                "our_answer_length": answer_length(answer),
                "our_sources": [
                    f"{s['source']} 第{s['page']}頁"
                    for s in result["sources"][:3]
                ],
            }
            self.results.append(evaluation)
            print(f"   覆蓋率: {evaluation['our_keyword_coverage']:.2%}")
            print(f"   有引用: {evaluation['our_has_citation']}")

        return self.results

    def load_notebooklm_answers(self, json_path: str = "./evaluation/notebooklm_answers.json"):
        """
        載入 NotebookLM 的答案 (需要你先到 NotebookLM 跑出來,存成 JSON)
        
        格式:
        [
          {"id": 1, "answer": "NotebookLM 的回答..."},
          ...
        ]
        """
        if not os.path.exists(json_path):
            print(f"⚠️ NotebookLM 答案檔案不存在: {json_path}")
            print("   請先到 https://notebooklm.google.com 上傳相同的 PDF,")
            print("   詢問相同題目後,把答案複製到 JSON 檔案中")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            nlm_answers = json.load(f)

        nlm_map = {item["id"]: item["answer"] for item in nlm_answers}

        # 把 NotebookLM 答案加入評估
        for result in self.results:
            nlm_ans = nlm_map.get(result["id"], "")
            result["notebooklm_answer"] = nlm_ans
            result["notebooklm_keyword_coverage"] = keyword_coverage(
                nlm_ans, result["ground_truth"]
            )
            result["notebooklm_has_citation"] = has_citation(nlm_ans)
            result["notebooklm_answer_length"] = answer_length(nlm_ans)

        print(f"✅ 已載入 NotebookLM {len(nlm_answers)} 題答案")

    def generate_report(self, output_dir: str = "./evaluation"):
        """產出分析報告"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 存成 CSV
        df = pd.DataFrame(self.results)
        csv_path = f"{output_dir}/evaluation_results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n📊 結果儲存於: {csv_path}")

        # 2. 統計指標
        print("\n" + "=" * 60)
        print("📈 整體統計")
        print("=" * 60)
        print(f"題目總數: {len(df)}")
        print(f"\n【我們的系統】")
        print(f"  平均關鍵字覆蓋率: {df['our_keyword_coverage'].mean():.2%}")
        print(f"  含引用比例: {df['our_has_citation'].mean():.2%}")
        print(f"  平均回答長度: {df['our_answer_length'].mean():.0f} 字")

        if "notebooklm_keyword_coverage" in df.columns:
            print(f"\n【NotebookLM】")
            print(f"  平均關鍵字覆蓋率: {df['notebooklm_keyword_coverage'].mean():.2%}")
            print(f"  含引用比例: {df['notebooklm_has_citation'].mean():.2%}")
            print(f"  平均回答長度: {df['notebooklm_answer_length'].mean():.0f} 字")

        # 3. 分難度統計
        print(f"\n【依難度區分】")
        diff_stats = df.groupby("difficulty")["our_keyword_coverage"].mean()
        for diff, score in diff_stats.items():
            print(f"  {diff}: {score:.2%}")

        # 4. 產出 Markdown 報告
        self._generate_markdown_report(df, output_dir)

    def _generate_markdown_report(self, df, output_dir):
        """產生 Markdown 格式的詳細比較報告"""
        md_path = f"{output_dir}/comparison_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 🔍 RAG 系統 vs NotebookLM 比較分析報告\n\n")
            f.write(f"產出時間:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # 統計摘要
            f.write("## 📊 整體效能比較\n\n")
            f.write("| 指標 | 我們的系統 | NotebookLM |\n")
            f.write("|------|----------|----------|\n")
            f.write(
                f"| 平均關鍵字覆蓋率 | {df['our_keyword_coverage'].mean():.2%} | "
            )
            if "notebooklm_keyword_coverage" in df.columns:
                f.write(f"{df['notebooklm_keyword_coverage'].mean():.2%} |\n")
            else:
                f.write("(未提供) |\n")

            f.write(
                f"| 含引用條款比例 | {df['our_has_citation'].mean():.2%} | "
            )
            if "notebooklm_has_citation" in df.columns:
                f.write(f"{df['notebooklm_has_citation'].mean():.2%} |\n")
            else:
                f.write("(未提供) |\n")

            f.write("\n---\n\n")

            # 逐題比較
            f.write("## 🧪 逐題比較\n\n")
            for r in self.results:
                f.write(f"### Q{r['id']}: {r['question']}\n")
                f.write(f"**類別**: {r['category']} | **難度**: {r['difficulty']}\n\n")
                f.write(f"**Ground Truth**: {r['ground_truth']}\n\n")

                f.write("#### 🔵 我們的系統\n")
                f.write(f"{r['our_answer']}\n\n")
                f.write(f"- 關鍵字覆蓋率: {r['our_keyword_coverage']:.2%}\n")
                f.write(f"- 含引用: {'✅' if r['our_has_citation'] else '❌'}\n")
                f.write(f"- 引用來源: {', '.join(r['our_sources'])}\n\n")

                if "notebooklm_answer" in r:
                    f.write("#### 🟢 NotebookLM\n")
                    f.write(f"{r['notebooklm_answer']}\n\n")
                    f.write(f"- 關鍵字覆蓋率: {r['notebooklm_keyword_coverage']:.2%}\n")
                    f.write(f"- 含引用: {'✅' if r['notebooklm_has_citation'] else '❌'}\n\n")

                f.write("---\n\n")

            # 分析結論 (留空白讓你自己寫)
            f.write("## 🎯 分析結論 (自填)\n\n")
            f.write("### 我們的系統優勢\n")
            f.write("- (例) 引用條款明確,可溯源\n")
            f.write("- (例) 中文語意捕捉精準\n\n")
            f.write("### 我們的系統弱勢\n")
            f.write("- (例) 在多層條件推理上偶爾失準\n\n")
            f.write("### NotebookLM 優勢\n")
            f.write("- (例) 回答流暢度高\n\n")
            f.write("### 改進方向\n")
            f.write("- (例) 增加 Chunk Size 改善上下文\n")
            f.write("- (例) 加入 Reranker 提升檢索品質\n")

        print(f"📝 詳細報告儲存於: {md_path}")


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_our_system()
    evaluator.load_notebooklm_answers()
    evaluator.generate_report()
