import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_text
from src.rag.semantic_index import SemanticIndex

DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_test_cases.json"
TOP_K = 3
MIN_SCORE = 0.20


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0

    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0

    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def load_cases() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    return cases


def evaluate_case(case: dict, index: SemanticIndex) -> dict:
    chunks = []

    for document in case["documents"]:
        chunks.extend(
            chunk_text(
                text=document["text"],
                source_id=document["source_id"],
                source_type="evaluation",
                chunk_size=100,
                overlap=0,
            )
        )

    relevant_ids = set(case["relevant_source_ids"])
    available_ids = {chunk.source_id for chunk in chunks}

    if relevant_ids and not relevant_ids.issubset(available_ids):
        raise ValueError(
            f"{case['case_id']}: a relevant source ID is missing from documents."
        )

    index.fit(chunks)
    results = index.search(
    case["query"],
    top_k=TOP_K,
    min_score=MIN_SCORE,
    )
    retrieved_ids = [result.chunk.source_id for result in results]

    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "relevant_ids": sorted(relevant_ids),
        "retrieved_ids": retrieved_ids,
        "scores": [round(result.score, 4) for result in results],
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, TOP_K),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        "no_answer_case": not relevant_ids,
        "false_positive_no_answer": not relevant_ids and bool(retrieved_ids),
    }


def main() -> None:
    cases = load_cases()

    # Load the embedding model once; reuse it across all cases.
    index = SemanticIndex()
    results = [evaluate_case(case, index) for case in cases]

    print("\nSEMANTIC RETRIEVAL EVALUATION")
    print("=" * 72)

    for result in results:
        print(f"\n{result['case_id']}: {result['query']}")
        print(f"  Expected:  {result['relevant_ids']}")
        print(f"  Retrieved: {result['retrieved_ids']}")
        print(f"  Scores:    {result['scores']}")
        print(f"  Recall@3:  {result['recall_at_3']:.2f}")
        print(f"  RR:        {result['reciprocal_rank']:.2f}")

        if result["false_positive_no_answer"]:
            print("  WARNING: returned results for a no-answer case")

    recall_scores = [r["recall_at_3"] for r in results]
    reciprocal_ranks = [r["reciprocal_rank"] for r in results]
    no_answer_cases = [r for r in results if r["no_answer_case"]]
    false_positives = sum(r["false_positive_no_answer"] for r in results)

    print("\n" + "=" * 72)
    print(f"Cases evaluated: {len(results)}")
    print(f"Average Recall@3: {sum(recall_scores) / len(results):.3f}")
    print(f"Mean Reciprocal Rank (MRR): {sum(reciprocal_ranks) / len(results):.3f}")
    print(f"No-answer cases: {len(no_answer_cases)}")
    print(f"No-answer cases returning results: {false_positives}")
    print("\nNote: this is a small synthetic dataset, not a production benchmark.")


if __name__ == "__main__":
    main()

