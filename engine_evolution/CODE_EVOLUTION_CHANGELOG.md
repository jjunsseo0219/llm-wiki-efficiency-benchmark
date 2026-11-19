# AI 위키 검색 엔진(`rag_engine.py`) 실험별 코드 진화 일지

> **디렉터리 위치:** `engine_evolution/`
> **포함 파일:**
> - [`rag_engine_git_head.py`](./rag_engine_git_head.py) : Git HEAD 기준 초기 원본 (21,876 bytes)
> - [`rag_engine_v4_final.py`](./rag_engine_v4_final.py) : 4대 패치 완료 최종본 (47,483 bytes)
> - [`rag_engine_head_to_v4.patch`](./rag_engine_head_to_v4.patch) : Git HEAD 대비 전체 Diff 패치 파일 (42,393 bytes)

---

## 📌 실험 버전별 핵심 코드 변경 요약

### 1. v1 (1호 패치 적용 — 1차 실험 기준)
* **진단 문제:** `_전체_대시보드.md`, `_프로젝트_지도.md` 등 모든 키워드가 모인 목차 문서가 본문을 밀어내고 검색 1위를 독식함.
* **코드 패치 (`_is_index_hub` 1차 도입):**
  * `os.path.basename(filepath).startswith("_")` 조건으로 목차 허브 문서를 `scan_all_notes` 및 `hybrid_search_notes`에서 배제.
* **당시 성적 (1-A 20문항):** `Hit@1` 80.0% / `MRR` 0.858 / `Cov@5` 73.3% (L3: 36.1%) / `651.60 ms`
* **잔존 결함 발견:** DB 내 `.raw`(89개)·`.archive`(74개) 청크가 잔존해 상위 슬롯을 3중 도배 중이었고, `get_cached_chunk_matrix()`의 `len(emb) == 1024` 하드코딩으로 인해 768차원(`embeddinggemma-2`) 벡터 검색이 마비 상태였음.

---

### 2. v2 (2·3호 패치 적용 — 좀비 청소 및 768차원 벡터 캐시 복구)
* **코드 패치 A (`_is_index_hub` 확장 및 DB 영구 청소):**
  ```python
  def _is_index_hub(filepath: str) -> bool:
      """목차/대시보드 및 비활성 폴더(.raw, .archive, 대화기록) 문서를 검색·색인에서 제외합니다."""
      norm = filepath.replace("\\", "/")
      parts = norm.split("/")
      if any(p in (".raw", ".archive", "대화기록") for p in parts):
          return True
      return os.path.basename(filepath).startswith("_")
  ```
  * `queue.db` 내 좀비 레코드 163개 파일 삭제 (`248개 파일 4,236청크` $\rightarrow$ **`79개 활성 문서 1,354청크`**).
* **코드 패치 B (`get_cached_chunk_matrix` 동적 차원 감지):**
  ```python
  # [Before] len(emb) == 1024 하드코딩으로 인해 768차원 청크 전량 폐기
  # if len(emb) == 1024: ...

  # [After] 최빈 차원(768d) 자동 감지로 어떤 임베딩 모델도 호환
  target_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
  chunk_embeddings_list = [emb for emb, _ in parsed_rows if len(emb) == target_dim]
  ```
* **성능 변화 (1-A 20문항):** `Hit@1` **90.0%** / `MRR` **0.950** / `Cov@5` **86.7%** / **`18.54 ms`** (35배 가속).

---

### 3. v3 (실험 1-B 투입 시 발견된 4호 결함 상태)
* **진단 문제:** 고유명사·제목을 뺀 긴 자연어 질문(1-B)을 던졌을 때, 흔한 단어(`학교`, `학생`, `시스템`)의 점수 누적으로 인해 `match_score >= 12.0` 조건이 오발동하여 **20문항 중 17문항에서 벡터 검색을 스킵하고 조기 종료**됨.
* **당시 성적 (1-B 20문항):** `Hit@1` 60.0% / `MRR` 0.750 / `Cov@5` 74.2% (BM25 0.798 대비 열세).

---

### 4. v4 (4호 패치 적용 — 최종 완전체)
* **코드 패치 (`hybrid_search_notes` Fast-Path 조건 엄격화):**
  ```python
  # [Before] 긴 문장도 키워드 누적 점수 >= 12.0이면 벡터 검색 생략
  # if top_hit.get("match_score", 0) >= 12.0: ...

  # [After] 3단어 이하 짧은 질의 또는 제목 완전 일치일 때만 Fast-Path 허용
  q_clean = query.strip().lower()
  if fast_path and kw_results:
      top_hit = kw_results[0]
      is_short_query = len(q_clean.split()) <= 3
      is_exact_title = q_clean in top_hit.get("title", "").lower()
      if (is_short_query or is_exact_title) and top_hit.get("match_score", 0) >= 12.0:
          ...
  ```
* **최종 성적:**
  * **실험 1-A (일반 20문항):** `Hit@1` **95.0%** / `Hit@3` **100%** / `MRR` **0.975** / `Cov@5` **94.2%** (`166.37 ms`)
  * **실험 1-B (모호한 20문항):** `Hit@1` **80.0%** (L3: **83.3%**) / `Hit@3` **95.0%** / `MRR` **0.875** (`148.96 ms`)
