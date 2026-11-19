import os
import sys
import json
import time
import math
import re
import csv
import pathlib
from collections import Counter

# Setup paths & environment for Library B (AI Wiki)
os.environ["NOTES_DIR"] = r"C:\local wiki"
os.environ["USE_GEMINI_EMBEDDING"] = "0"
os.environ["EMBEDDING_DEVICE"] = "cpu"
sys.path.insert(0, r"E:\local_knowledge_log")

import rag_engine

EXP_DIR = pathlib.Path(r"c:\학교 내신 관리\experiment")
LIB_A_DIR = EXP_DIR / "library_a_raw"
RES_DIR = EXP_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_ko(text: str) -> list[str]:
    """공정한 한국어 키워드 검색(BM25)을 위한 어절 + 2-gram 복합 토크나이저."""
    clean = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text.lower())
    words = clean.split()
    tokens = []
    for w in words:
        tokens.append(w)
        if len(w) >= 2:
            # 조사/어미 변화에도 키워드가 걸리도록 2글자 이상 바이그램 보강
            for i in range(len(w) - 1):
                tokens.append(w[i:i+2])
    return tokens


class BM25Searcher:
    """서재 A(기성 폴더 방식)용 표준 Okapi BM25 검색기."""
    def __init__(self, doc_dir: pathlib.Path, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_tokens = []
        self.doc_lens = []
        self.df = Counter()

        for p in sorted(doc_dir.glob("*.md")):
            text = p.read_text(encoding="utf-8", errors="ignore")
            # 파일명도 검색 대상에 포함 (실제 파일 탐색기/폴더 검색과 동일한 조건)
            full_text = p.stem.replace("_", " ") + " " + text
            toks = tokenize_ko(full_text)
            self.docs.append({"name": p.name, "path": str(p)})
            self.doc_tokens.append(Counter(toks))
            self.doc_lens.append(len(toks))
            for t in set(toks):
                self.df[t] += 1

        self.N = len(self.docs)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1.0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_toks = tokenize_ko(query)
        scores = []
        for idx in range(self.N):
            score = 0.0
            dl = self.doc_lens[idx]
            tf_map = self.doc_tokens[idx]
            for qt in q_toks:
                if qt not in tf_map:
                    continue
                df = self.df[qt]
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map[qt]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (num / den)
            scores.append((score, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"name": self.docs[i]["name"], "score": round(s, 4)} for s, i in scores[:top_k]]


def eval_targets(retrieved_names: list[str], target_patterns: list[str]) -> dict:
    """
    상위 5개 결과에 대한 Hit@1, Hit@3, Hit@5, MRR, Multi-hop Target Coverage@5 계산.
    각 target_pattern이 독립적인 필수 근거 문서(Hop)에 해당.
    """
    hit1 = 0
    hit3 = 0
    hit5 = 0
    first_rank = None

    for rank_idx, fname in enumerate(retrieved_names[:5], start=1):
        if any(pat in fname for pat in target_patterns):
            if first_rank is None:
                first_rank = rank_idx
            if rank_idx == 1:
                hit1 = 1
            if rank_idx <= 3:
                hit3 = 1
            hit5 = 1

    mrr = (1.0 / first_rank) if first_rank else 0.0

    # Multi-hop Coverage: 필요한 서로 다른 타겟 문서들 중 몇 개나 상위 5위 내에 포함되었는가
    found_targets = 0
    for pat in target_patterns:
        if any(pat in fname for fname in retrieved_names[:5]):
            found_targets += 1
    coverage5 = found_targets / len(target_patterns) if target_patterns else 0.0

    return {
        "hit1": hit1,
        "hit3": hit3,
        "hit5": hit5,
        "mrr": round(mrr, 4),
        "coverage5": round(coverage5, 4),
        "found_targets": found_targets,
        "total_targets": len(target_patterns),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--prefix", default="retrieval")
    args = parser.parse_args()

    questions = json.loads((EXP_DIR / args.questions).read_text(encoding="utf-8"))

    print("1) Initializing Library A (BM25 over 89 raw files)...")
    t0 = time.perf_counter()
    bm25 = BM25Searcher(LIB_A_DIR)
    init_a_ms = (time.perf_counter() - t0) * 1000
    print(f"   Library A ready: {bm25.N} docs in {init_a_ms:.1f} ms")

    print("2) Warming up Library B (AI Wiki True RRF + embeddinggemma-2)...")
    t0 = time.perf_counter()
    rag_engine.get_cached_chunk_matrix(force_reload=False)
    import embedder
    embedder.get_embedding_model()
    # 1회 더미 쿼리로 토크나이저 및 행렬 연산 워밍업
    _ = rag_engine.hybrid_search_notes("워밍업 테스트", top_k=5, fast_path=True)
    init_b_ms = (time.perf_counter() - t0) * 1000
    print(f"   Library B ready in {init_b_ms:.1f} ms")

    rows = []
    for q in questions:
        qid = q["id"]
        lvl = q["level"]
        query = q["query"]

        # Test Library A (BM25)
        ta0 = time.perf_counter()
        res_a = bm25.search(query, top_k=5)
        lat_a_ms = (time.perf_counter() - ta0) * 1000
        names_a = [r["name"] for r in res_a]
        eval_a = eval_targets(names_a, q["raw_keywords"])

        # Test Library B (AI Wiki RRF)
        tb0 = time.perf_counter()
        res_b = rag_engine.hybrid_search_notes(query, top_k=5, fast_path=True)
        lat_b_ms = (time.perf_counter() - tb0) * 1000
        names_b = [os.path.basename(r["path"]) for r in res_b]
        eval_b = eval_targets(names_b, q["wiki_targets"])

        rows.append({
            "id": qid,
            "level": lvl,
            "hop": q["hop"],
            "a_hit1": eval_a["hit1"],
            "a_hit3": eval_a["hit3"],
            "a_hit5": eval_a["hit5"],
            "a_mrr": eval_a["mrr"],
            "a_cov5": eval_a["coverage5"],
            "a_lat_ms": round(lat_a_ms, 2),
            "b_hit1": eval_b["hit1"],
            "b_hit3": eval_b["hit3"],
            "b_hit5": eval_b["hit5"],
            "b_mrr": eval_b["mrr"],
            "b_cov5": eval_b["coverage5"],
            "b_lat_ms": round(lat_b_ms, 2),
            "a_top3": " | ".join(names_a[:3]),
            "b_top3": " | ".join(names_b[:3]),
        })
        print(f"[{qid} L{lvl}] A(cov={eval_a['coverage5']:.2f}, mrr={eval_a['mrr']:.2f}, {lat_a_ms:.1f}ms) vs B(cov={eval_b['coverage5']:.2f}, mrr={eval_b['mrr']:.2f}, {lat_b_ms:.1f}ms)")

    # Save CSV
    csv_path = RES_DIR / f"{args.prefix}_detail.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate Summary by Level & Overall
    summary = {"init_a_ms": round(init_a_ms, 2), "init_b_ms": round(init_b_ms, 2), "by_level": {}, "overall": {}}
    for lvl in [1, 2, 3]:
        sub = [r for r in rows if r["level"] == lvl]
        n = len(sub)
        summary["by_level"][f"L{lvl}"] = {
            "n": n,
            "A": {
                "hit1": round(sum(r["a_hit1"] for r in sub) / n, 4),
                "hit3": round(sum(r["a_hit3"] for r in sub) / n, 4),
                "hit5": round(sum(r["a_hit5"] for r in sub) / n, 4),
                "mrr": round(sum(r["a_mrr"] for r in sub) / n, 4),
                "cov5": round(sum(r["a_cov5"] for r in sub) / n, 4),
                "lat_ms": round(sum(r["a_lat_ms"] for r in sub) / n, 2),
            },
            "B": {
                "hit1": round(sum(r["b_hit1"] for r in sub) / n, 4),
                "hit3": round(sum(r["b_hit3"] for r in sub) / n, 4),
                "hit5": round(sum(r["b_hit5"] for r in sub) / n, 4),
                "mrr": round(sum(r["b_mrr"] for r in sub) / n, 4),
                "cov5": round(sum(r["b_cov5"] for r in sub) / n, 4),
                "lat_ms": round(sum(r["b_lat_ms"] for r in sub) / n, 2),
            },
        }

    n_all = len(rows)
    summary["overall"] = {
        "n": n_all,
        "A": {
            "hit1": round(sum(r["a_hit1"] for r in rows) / n_all, 4),
            "hit3": round(sum(r["a_hit3"] for r in rows) / n_all, 4),
            "hit5": round(sum(r["a_hit5"] for r in rows) / n_all, 4),
            "mrr": round(sum(r["a_mrr"] for r in rows) / n_all, 4),
            "cov5": round(sum(r["a_cov5"] for r in rows) / n_all, 4),
            "lat_ms": round(sum(r["a_lat_ms"] for r in rows) / n_all, 2),
        },
        "B": {
            "hit1": round(sum(r["b_hit1"] for r in rows) / n_all, 4),
            "hit3": round(sum(r["b_hit3"] for r in rows) / n_all, 4),
            "hit5": round(sum(r["b_hit5"] for r in rows) / n_all, 4),
            "mrr": round(sum(r["b_mrr"] for r in rows) / n_all, 4),
            "cov5": round(sum(r["b_cov5"] for r in rows) / n_all, 4),
            "lat_ms": round(sum(r["b_lat_ms"] for r in rows) / n_all, 2),
        },
    }

    (RES_DIR / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
