"""Evaluation harness.

    python -m eval.run_eval                 # retrieval metrics only, no LLM calls
    python -m eval.run_eval --with-answers  # also generates answers (costs tokens)
    python -m eval.run_eval --top-k 5       # sweep K to see the recall/precision trade

Retrieval is evaluated separately from generation on purpose. When an answer
is wrong, the first question is always whether the right chunk was retrieved.
Measuring only end-to-end answer quality hides which half is broken.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.core.prompt import REFUSAL  # noqa: E402
from app.services.dependencies import get_retriever, get_vector_store  # noqa: E402
from app.services.rag_service import answer_question  # noqa: E402
from eval.metrics import aggregate, first_relevant_rank, hit, reciprocal_rank  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_golden(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_retrieval(questions: list, top_k: int) -> list:
    retriever = get_retriever()
    rows = []

    for item in questions:
        chunks = retriever.retrieve_unfiltered(item["question"], top_k=top_k)
        pages = [chunk["page"] for chunk in chunks]
        expected = item.get("expected_pages", [])

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_pages": expected,
                "retrieved_pages": pages,
                "top_distance": round(chunks[0]["distance"], 4) if chunks else None,
                "rank": first_relevant_rank(pages, expected),
                "hit": hit(pages, expected),
                "reciprocal_rank": round(reciprocal_rank(pages, expected), 3),
            }
        )

    return rows


def evaluate_answers(questions: list, top_k: int) -> list:
    rows = []

    for item in questions:
        result = answer_question(item["question"], top_k=top_k)
        answer_lower = result["answer"].lower()
        keywords = item.get("expected_keywords", [])
        found = [k for k in keywords if k.lower() in answer_lower]

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "grounded": result["grounded"],
                "keywords_expected": keywords,
                "keywords_found": found,
                "keyword_pass": len(found) == len(keywords) if keywords else None,
                "latency_ms": result["latency_ms"],
                "answer": result["answer"],
            }
        )

    return rows


def evaluate_refusals(items: list, top_k: int) -> list:
    """Out-of-scope questions should be refused, not answered."""
    rows = []

    for item in items:
        result = answer_question(item["question"], top_k=top_k)
        refused = (not result["grounded"]) or result["answer"].strip() == REFUSAL
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "refused": refused,
                "answer": result["answer"][:300],
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--golden", default=str(EVAL_DIR / "golden_set.json"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--with-answers", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.top_k

    if get_vector_store().count() == 0:
        raise SystemExit("Vector store is empty. Ingest a PDF before running eval.")

    golden = load_golden(Path(args.golden))
    questions = golden["questions"]

    print(f"\nRetrieval evaluation  |  top_k={top_k}  |  {len(questions)} questions\n")
    retrieval_rows = evaluate_retrieval(questions, top_k)

    print(f"{'id':<5} {'hit':<5} {'rank':<5} {'expected':<16} retrieved")
    for row in retrieval_rows:
        print(
            f"{row['id']:<5} {str(row['hit']):<5} {row['rank']:<5} "
            f"{str(row['expected_pages']):<16} {row['retrieved_pages']}"
        )

    summary = aggregate(retrieval_rows)
    print(f"\n  hit rate : {summary['hit_rate']}")
    print(f"  MRR      : {summary['mrr']}")

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "generation_model": settings.generation_model,
            "max_distance": settings.max_distance,
        },
        "retrieval": {"summary": summary, "rows": retrieval_rows},
    }

    if args.with_answers:
        print("\nGenerating answers...\n")
        answer_rows = evaluate_answers(questions, top_k)
        passed = [r for r in answer_rows if r["keyword_pass"]]
        print(f"  keyword pass : {len(passed)}/{len(answer_rows)}")

        refusal_rows = evaluate_refusals(golden.get("out_of_scope", []), top_k)
        refused = [r for r in refusal_rows if r["refused"]]
        print(f"  refusal rate : {len(refused)}/{len(refusal_rows)} out-of-scope refused")

        report["answers"] = answer_rows
        report["refusals"] = refusal_rows

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"eval-{stamp}-k{top_k}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved: {out_path}\n")


if __name__ == "__main__":
    main()
