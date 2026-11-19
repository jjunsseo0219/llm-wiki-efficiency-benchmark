import pathlib
import re
import json


def main():
    raw_dir = pathlib.Path(r"c:\학교 내신 관리\experiment\library_a_raw")
    wiki_dir = pathlib.Path(r"C:\local wiki")

    # 1. Connectivity (Orphan Note Ratio)
    raw_files = list(raw_dir.glob("*.md"))
    raw_linked = sum(
        1 for p in raw_files
        if re.search(r"\[\[[^\]]+\]\]", p.read_text(encoding="utf-8", errors="ignore"))
    )
    raw_orphan = len(raw_files) - raw_linked

    w_files = [
        p for p in wiki_dir.rglob("*.md")
        if not any(pt.startswith(".") for pt in p.relative_to(wiki_dir).parts)
        and not p.name.startswith("_")
    ]
    out_links = {}
    in_counts = {p.stem: 0 for p in w_files}
    for p in w_files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        links = re.findall(r"\[\[([^\]|]+)", txt)
        out_links[p.stem] = len(links)
        for lk in links:
            lk_clean = pathlib.Path(lk).stem
            for target_stem in in_counts:
                if lk_clean == target_stem or (len(lk_clean) >= 4 and lk_clean in target_stem):
                    in_counts[target_stem] += 1

    w_connected = sum(1 for p in w_files if out_links[p.stem] > 0 or in_counts[p.stem] > 0)
    w_orphan = len(w_files) - w_connected

    # 2. Fidelity (Fact/Number Preservation across raw_source-linked pairs)
    pairs = []
    for p in w_files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'raw_source:\s*"([^"]+)"', txt)
        if m:
            raw_path = raw_dir / m.group(1)
            if raw_path.exists():
                pairs.append((raw_path, p))

    num_pat = re.compile(r"\d+(?:[.,]\d+)?(?:\s*%|\s*만\s*원|\s*억|\s*KB|\s*GB|\s*ms|\s*nm|\s*pm|\s*GeV|\s*Gy)")
    total_raw_facts = 0
    preserved_facts = 0
    pair_scores = []
    for rp, wp in pairs:
        rtxt = rp.read_text(encoding="utf-8", errors="ignore")
        wtxt = wp.read_text(encoding="utf-8", errors="ignore")
        rfacts = set(re.sub(r"\s+", "", x) for x in num_pat.findall(rtxt))
        if not rfacts:
            continue
        wnorm = re.sub(r"\s+", "", wtxt)
        kept = sum(1 for f in rfacts if f in wnorm)
        total_raw_facts += len(rfacts)
        preserved_facts += kept
        pair_scores.append(kept / len(rfacts))

    res = {
        "connectivity": {
            "raw_total": len(raw_files),
            "raw_linked": raw_linked,
            "raw_orphan": raw_orphan,
            "raw_orphan_pct": round(raw_orphan / len(raw_files) * 100, 1),
            "wiki_total": len(w_files),
            "wiki_connected": w_connected,
            "wiki_orphan": w_orphan,
            "wiki_orphan_pct": round(w_orphan / len(w_files) * 100, 1),
            "wiki_connected_pct": round(w_connected / len(w_files) * 100, 1),
            "total_wiki_out_links": sum(out_links.values()),
        },
        "fidelity": {
            "matched_pairs": len(pairs),
            "evaluated_pairs_with_numbers": len(pair_scores),
            "total_raw_numeric_facts": total_raw_facts,
            "preserved_numeric_facts": preserved_facts,
            "micro_preservation_pct": round(preserved_facts / total_raw_facts * 100, 1) if total_raw_facts else 0,
            "macro_avg_pct": round(sum(pair_scores) / len(pair_scores) * 100, 1) if pair_scores else 0,
        },
    }
    out_p = pathlib.Path(r"c:\학교 내신 관리\experiment\results\connectivity_and_fidelity.json")
    out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
