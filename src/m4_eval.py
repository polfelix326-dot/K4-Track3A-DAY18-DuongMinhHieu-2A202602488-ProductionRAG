from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import math
    zeros = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }
    if not questions or not answers or not contexts or not ground_truths:
        return zeros

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        def _safe_val(v):
            if v is None:
                return 0.0
            if isinstance(v, float) and math.isnan(v):
                return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0

        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_val(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_val(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_val(row.get("context_precision", 0.0)),
                context_recall=_safe_val(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        def _metric_avg(col_name):
            if hasattr(result, "get") and result.get(col_name) is not None:
                return _safe_val(result.get(col_name))
            if col_name in df:
                return _safe_val(df[col_name].mean())
            return 0.0

        return {
            "faithfulness": _metric_avg("faithfulness"),
            "answer_relevancy": _metric_avg("answer_relevancy"),
            "context_precision": _metric_avg("context_precision"),
            "context_recall": _metric_avg("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu (LLM hallucinating)",
            "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng (Missing relevant chunks)",
            "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25",
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu (Too many irrelevant chunks)",
            "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata",
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi (Answer doesn't match question)",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn",
        ),
    }

    analyses = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric,
            ("Unknown issue", "Investigate manually")
        )
        analyses.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyses.sort(key=lambda x: x["avg_score"])
    return analyses[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
