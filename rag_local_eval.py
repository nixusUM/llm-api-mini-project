import argparse
import json
from pathlib import Path

from rag_service import RAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG vs cloud generation.")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per backend/question.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/pipeline_outputs/rag_local_vs_cloud_report.json",
        help="Where to save JSON report.",
    )
    parser.add_argument("--top-k-before", type=int, default=None)
    parser.add_argument("--top-k-after", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RAGService()
    report = service.compare_generation_backends(
        repeats=max(1, int(args.repeats)),
        top_k_before=args.top_k_before,
        top_k_after=args.top_k_after,
        threshold=args.threshold,
        enable_query_rewrite=True,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved report: {output_path}")
    print(f"Questions: {report.get('total_questions', 0)}")
    print(f"Local quality OK: {report.get('local_quality_ok', 0)}")
    print(f"Cloud quality OK: {report.get('cloud_quality_ok', 0)}")


if __name__ == "__main__":
    main()
