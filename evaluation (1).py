import difflib
import logging
import os
import json
import sqlglot
import time
from query_chain import create_query_chain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ground truth test cases (optional SQL for comparison)
TEST_CASES = [
    {
        "query": "Show all consultants",
        "expected_sql": "SELECT * FROM HCP WHERE isconsultant = TRUE;"
    },
    {
        "query": "List medical reps and their HCP names",
        "expected_sql": "SELECT MRId, HCPEnglishName FROM MedicalReps;"
    },
    {
        "query": "Which HCPs are decision makers in Egypt?",
        "expected_sql": "SELECT englishname FROM HCP WHERE isdecisionmaker = TRUE AND Country = 'Egypt';"
    },
    {
        "query": "Find all HCPs who are university staff",
        "expected_sql": "SELECT * FROM HCP WHERE isuniversitystaff = TRUE;"
    },
    {
        "query": "Get consultants along with their specialties",
        "expected_sql": "SELECT englishname, Speciality FROM HCP WHERE isconsultant = TRUE;"
    }
]


def normalize_sql(sql: str) -> str:
    """Normalize SQL using sqlglot for fair comparison."""
    try:
        return sqlglot.parse_one(sql).sql(dialect="ansi").lower()
    except Exception:
        # fallback: lowercase and strip formatting
        return " ".join(sql.replace("\n", " ").split()).lower()


def sql_similarity(a: str, b: str) -> float:
    """Compute similarity score between two SQL queries after normalization."""
    a_norm = normalize_sql(a)
    b_norm = normalize_sql(b)
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def run_evaluation(output_file: str = "evaluation_GEMINI_results.json"):
    logger.info("Initializing Query-to-SQL chain for evaluation...")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    query_chain = create_query_chain(top_k=2, persist_directory=persist_dir)
    logger.info("Chain initialized successfully ✅")

    results = []

    for case in TEST_CASES:
        query = case["query"]
        expected_sql = case.get("expected_sql")

        print("\n" + "=" * 70)
        print(f"User Query: {query}")

        # Measure response time
        start_time = time.time()
        detailed = query_chain.get_detailed_response(query)
        elapsed_time = time.time() - start_time

        print("\n--- Detailed Response ---")
        for k, v in detailed.items():
            print(f"{k}: {v}")

        generated_sql = detailed.get("sql_query", "N/A")

        similarity = None
        exact_match = False
        if expected_sql:
            similarity = sql_similarity(generated_sql, expected_sql)

            # Check for exact match after normalization
            exact_match = normalize_sql(generated_sql) == normalize_sql(expected_sql)

            print(f"\nExpected SQL: {expected_sql}")
            print(f"Generated SQL: {generated_sql}")
            print(f"Similarity Score: {similarity:.2f}")
            print(f"Exact Match: {exact_match}")
        else:
            print("\n(No ground truth SQL provided)")

        print(f"Response Time: {elapsed_time:.2f}s")

        results.append({
            "query": query,
            "expected_sql": expected_sql,
            "generated_sql": generated_sql,
            "similarity": similarity,
            "exact_match": exact_match,
            "response_time": elapsed_time,
            "detailed": detailed
        })

    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY (saved to {output_file})")
    for r in results:
        print(f"- Query: {r['query']}")
        if r["expected_sql"]:
            print(f"  Similarity: {r['similarity']:.2f}")
            print(f"  Exact Match: {r['exact_match']}")
        else:
            print("  (No ground truth SQL provided)")
        print(f"  Response Time: {r['response_time']:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
