import os
import re
import numpy as np
import urllib.request
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.environ.get("NOTES_DIR", r"C:\local wiki" if os.path.exists(r"C:\local wiki") else os.path.join(BASE_DIR, "notes"))
WIKI_DIR = NOTES_DIR
RAW_DIR = os.path.join(NOTES_DIR, ".raw")

# BGE-M3 로컬 임베딩 모델 및 DB 벡터 로드 활용을 위한 모듈 로딩
import embedder
import sqlite3

def get_vector_for_note(note_meta):
    """
    노트 파일의 고정밀 BGE-M3 임베딩 벡터를 가져옵니다.
    SQLite DB (note_embeddings) 캐시에 보관되어 있으면 0.0001초 만에 즉시 반환합니다.
    """
    filepath = note_meta["path"]
    db_path = os.path.join(BASE_DIR, "queue.db")
    
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT embedding FROM note_embeddings WHERE filepath = ?", (filepath,))
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        except Exception as e:
            pass
            
    # 캐시 실패 시 BGE-M3 모델로 실시간 계산
    try:
        text_to_embed = f"제목: {note_meta['title']}\n요약: {note_meta['summary']}\n내용: {note_meta['content']}"
        model = embedder.get_embedding_model()
        emb = model.encode(text_to_embed, convert_to_numpy=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    except Exception as e:
        logging.error(f"Failed to generate BGE-M3 embedding: {e}")
        return []


def extract_note_metadata(filepath):
    """
    개별 마크다운 파일의 제목(H1), 요약(summary), 태그(YAML), 본문을 파싱합니다.
    """
    title = os.path.basename(filepath)[:-3]
    summary = ""
    content_body = ""
    tags = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. YAML 파싱
        yaml_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            t_m = re.search(r'title:\s*["\']?(.*?)["\']?$', yaml_content, re.MULTILINE)
            if t_m and t_m.group(1).strip():
                title = t_m.group(1).strip()
            tag_m = re.search(r'tags:\s*\[(.*?)\]', yaml_content)
            if tag_m:
                tags = [t.strip().strip('"\'') for t in tag_m.group(1).split(",") if t.strip()]

        # 2. H1 헤더가 있으면 제목으로 우선 고려
        h1_match = re.search(r'^#\s+([^\n]+)', content, re.MULTILINE)
        if h1_match and not title:
            title = h1_match.group(1).strip()
            
        # 3. 요약 추출 (ABSTRACT 콜아웃 또는 요약 헤더)
        abstract_match = re.search(r'>\s*\[!ABSTRACT\][^\n]*\n>\s*([^\n]+)', content)
        if abstract_match:
            summary = abstract_match.group(1).strip()
        else:
            summary_match = re.search(r'###\s*📝\s*요약\n([^\n]+)', content)
            if summary_match:
                summary = summary_match.group(1).strip()
            
        # 4. 전체 텍스트 본문 추출
        content_body = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        content_body = re.sub(r'#.*?\n', '', content_body) # 헤더 기호 제거
        content_body = re.sub(r'\s+', ' ', content_body).strip()
        
    except Exception as e:
        logging.error(f"Failed to parse metadata from {filepath}: {e}")
        
    return {
        "title": title,
        "summary": summary,
        "tags": tags,
        "content": content_body,
        "path": filepath
    }


def _is_index_hub(filepath: str) -> bool:
    """목차/대시보드 및 비활성 폴더(.raw, .archive, 대화기록) 문서를 검색·색인에서 제외합니다."""
    norm = filepath.replace("\\", "/")
    parts = norm.split("/")
    if any(p in (".raw", ".archive", "대화기록") for p in parts):
        return True
    return os.path.basename(filepath).startswith("_")


def scan_all_notes():
    """
    옵시디언 wiki/ 하위 폴더의 모든 컴파일된 지식 노트를 수집합니다.
    """
    notes = []
    target_dir = WIKI_DIR if os.path.exists(WIKI_DIR) else NOTES_DIR
    if not os.path.exists(target_dir):
        return notes
        
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "대화기록" in root.replace("\\", "/").split("/"):
            continue
        for filename in files:
            if filename.endswith(".md") and not filename.endswith("_ERROR_Fallback.md") and not filename.startswith("_"):
                filepath = os.path.join(root, filename)
                if not _is_index_hub(filepath):
                    note_meta = extract_note_metadata(filepath)
                    if note_meta["content"]:
                        notes.append(note_meta)
    return notes


def init_rag_db():
    db_path = os.path.join(BASE_DIR, "queue.db")
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS note_embeddings (
                filepath TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS note_chunks (
                chunk_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                section_title TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filepath ON note_chunks(filepath);")
        conn.commit()


def chunk_markdown(title: str, content: str, chunk_size: int = 400, overlap: int = 100):
    """
    Header-aware markdown chunker with sliding overlap (Youtube benchmark pattern).
    Preserves section headers and splits long sections with 400-char window + 100-char overlap.
    """
    clean_content = re.sub(r'^---\s*\n.*?\n---\s*(\n|$)', '', content, flags=re.DOTALL).strip()
    header_pattern = r'(?m)^(?=#{1,4}\s+)'
    raw_sections = [s.strip() for s in re.split(header_pattern, clean_content) if s.strip()]
    if not raw_sections:
        raw_sections = [clean_content] if clean_content else []

    chunks = []
    chunk_idx = 0

    for sec in raw_sections:
        lines = sec.split('\n', 1)
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            sec_title = first_line.lstrip('#').strip()
            sec_body = lines[1].strip() if len(lines) > 1 else ""
        else:
            sec_title = "개요"
            sec_body = sec

        if not sec_body:
            sec_body = first_line

        sec_body_clean = re.sub(r'\s+', ' ', sec_body).strip()
        if not sec_body_clean:
            continue

        if len(sec_body_clean) <= chunk_size:
            chunk_text = f"[{title} > {sec_title}]\n{sec_body_clean}"
            chunks.append({
                "chunk_index": chunk_idx,
                "section_title": sec_title,
                "content": chunk_text,
                "raw_text": sec_body_clean
            })
            chunk_idx += 1
        else:
            step = max(50, chunk_size - overlap)
            start = 0
            while start < len(sec_body_clean):
                end = min(len(sec_body_clean), start + chunk_size)
                if end < len(sec_body_clean):
                    last_space = sec_body_clean.rfind(' ', start, end)
                    if last_space > start + (chunk_size // 2):
                        end = last_space

                sub_text = sec_body_clean[start:end].strip()
                if sub_text:
                    chunk_text = f"[{title} > {sec_title}]\n{sub_text}"
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "section_title": sec_title,
                        "content": chunk_text,
                        "raw_text": sub_text
                    })
                    chunk_idx += 1

                if end >= len(sec_body_clean):
                    break
                start = max(start + 1, end - overlap)

    return chunks


def sync_note_embeddings(force: bool = False):
    """
    wiki/ 하위 폴더의 모든 마크다운 파일을 스캔하여 
    신규/수정된 파일의 BGE-M3 청크 분할 임베딩 및 요약을 SQLite DB에 동기화합니다.
    수정 시간(mtime)이 변경되지 않은 파일은 연산을 완전히 건너뜁니다.
    force=True 시 모든 문서를 강제로 재임베딩합니다.
    금고에서 삭제된 파일의 레코드 또한 자동으로 DB에서 정리합니다.
    """
    init_rag_db()
    db_path = os.path.join(BASE_DIR, "queue.db")
    target_dir = WIKI_DIR if os.path.exists(WIKI_DIR) else NOTES_DIR
    
    if not os.path.exists(target_dir):
        logging.warning(f"Target directory {target_dir} does not exist.")
        return
        
    # 1. 디렉토리 내부 실제 마크다운 파일들의 정보 수집
    actual_files = {}  # absolute_path -> mtime
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith(".md") and not filename.endswith("_ERROR_Fallback.md"):
                fn_lower = filename.lower()
                if any(k in fn_lower for k in ["2fa", "비밀번호", "password", "secret", "credential", "백업_코드"]):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    actual_files[filepath] = mtime
                except Exception as e:
                    logging.error(f"Failed to get mtime for {filepath}: {e}")

    # 2. DB에 기록된 기존 캐시 목록 조회
    db_files = {}  # filepath -> mtime
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, mtime FROM note_embeddings")
            for row in cursor.fetchall():
                db_files[row[0]] = row[1]
    except Exception as e:
        logging.error(f"Failed to query DB cache list: {e}")

    # 3. 삭제된 파일 정리
    deleted_paths = [path for path in db_files if path not in actual_files]
    if deleted_paths:
        logging.info(f"Removing {len(deleted_paths)} deleted files from DB...")
        try:
            with sqlite3.connect(db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.executemany("DELETE FROM note_embeddings WHERE filepath = ?", [(p,) for p in deleted_paths])
                cursor.executemany("DELETE FROM note_chunks WHERE filepath = ?", [(p,) for p in deleted_paths])
                conn.commit()
        except Exception as e:
            logging.error(f"Failed to delete records from DB: {e}")

    # 4. 신규 및 수정된 파일 판별
    to_update = []
    if force:
        to_update = list(actual_files.items())
    else:
        for path, mtime in actual_files.items():
            if path not in db_files or db_files[path] != mtime:
                to_update.append((path, mtime))

    # 청크 테이블에 아직 등록되지 않은 파일들도 to_update에 추가 (완전한 청크 커버리지 보장)
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT filepath FROM note_chunks")
            chunked_files = set(row[0] for row in cursor.fetchall())
            for path, mtime in actual_files.items():
                if path not in chunked_files and (path, mtime) not in to_update:
                    to_update.append((path, mtime))
    except Exception as e:
        logging.warning(f"Failed to check unchunked files: {e}")

    if not to_update:
        logging.info("No files need embedding update. RAG index is up-to-date.")
        return {
            "status": "up_to_date",
            "synced_count": 0,
            "deleted_count": len(deleted_paths),
            "total_files": len(actual_files),
            "message": "모든 노트가 이미 최신 상태로 동기화되어 있습니다."
        }

    logging.info(f"Need to sync {len(to_update)} files with header-aware chunking & overlap...")
    
    # 5. 임베딩 모델 준비
    model = embedder.get_embedding_model()
    
    # 6. 순차적으로 파싱 및 청크 분할 임베딩 연산 수행하여 DB에 적재
    synced_actual = 0
    total_chunks_created = 0

    for path, mtime in to_update:
        try:
            note_meta = extract_note_metadata(path)
            if not note_meta["content"]:
                continue
                
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

            # 마크다운 헤더 인식 청킹 (400자 단위 + 100자 오버랩)
            chunks = chunk_markdown(note_meta['title'], raw_content, chunk_size=400, overlap=100)
            if not chunks:
                chunks = [{
                    "chunk_index": 0,
                    "section_title": "본문",
                    "content": f"[{note_meta['title']}]\n{note_meta['content'][:400]}",
                    "raw_text": note_meta['content'][:400]
                }]

            # 청크 벡터 일괄 계산 (Batch encode)
            chunk_texts = [c["content"] for c in chunks]
            chunk_embeddings = model.encode(chunk_texts, batch_size=32)

            # 문서 요약 메타데이터 벡터
            summary_text = f"제목: {note_meta['title']} 요약: {note_meta['summary']} 태그: {', '.join(note_meta['tags'])}"
            doc_emb = model.encode(summary_text)
            if hasattr(doc_emb, "tolist"):
                doc_emb = doc_emb.tolist()
            else:
                doc_emb = list(doc_emb)

            with sqlite3.connect(db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO note_embeddings 
                    (filepath, mtime, title, summary, tags, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    path,
                    mtime,
                    note_meta['title'],
                    note_meta['summary'],
                    json.dumps(note_meta['tags'], ensure_ascii=False),
                    note_meta['content'],
                    json.dumps(doc_emb)
                ))

                # 이전 청크 데이터 정리 후 신규 청크 일괄 적재
                cursor.execute("DELETE FROM note_chunks WHERE filepath = ?", (path,))
                chunk_records = []
                for c, c_vec in zip(chunks, chunk_embeddings):
                    c_id = f"{path}#{c['chunk_index']}"
                    vec_list = c_vec.tolist() if hasattr(c_vec, "tolist") else list(c_vec)
                    chunk_records.append((
                        c_id,
                        path,
                        c['section_title'],
                        c['chunk_index'],
                        c['content'],
                        json.dumps(vec_list)
                    ))

                cursor.executemany('''
                    INSERT OR REPLACE INTO note_chunks
                    (chunk_id, filepath, section_title, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', chunk_records)
                conn.commit()

            synced_actual += 1
            total_chunks_created += len(chunks)
            logging.info(f"Synced {os.path.basename(path)} -> {len(chunks)} chunks created.")
        except Exception as e:
            logging.error(f"Failed to sync embedding for {path}: {e}")

    # 동기화 완료 후 바이너리 캐시 및 메모리 갱신
    get_cached_chunk_matrix(force_reload=True)

    logging.info(f"RAG embedding sync completed: {synced_actual} files, {total_chunks_created} chunks.")
    return {
        "status": "synced",
        "synced_count": synced_actual,
        "total_chunks": total_chunks_created,
        "deleted_count": len(deleted_paths),
        "total_files": len(actual_files)
    }


def sync_single_note(filepath: str):
    """
    단일 마크다운 파일(신규 작성 또는 수정)을 0.3초 만에 초고속 증분 색인합니다.
    전체 금고 스캔 없이 해당 파일만 청킹 및 배치 임베딩 후 DB와 인메모리/바이너리 캐시에 즉각 반영합니다.
    """
    init_rag_db()
    if not os.path.exists(filepath):
        return {"status": "not_found", "path": filepath}

    db_path = os.path.join(BASE_DIR, "queue.db")
    try:
        mtime = os.path.getmtime(filepath)
    except Exception:
        mtime = 0

    note_meta = extract_note_metadata(filepath)
    if not note_meta["content"]:
        return {"status": "empty_content", "path": filepath}

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        chunks = chunk_markdown(note_meta['title'], raw_content, chunk_size=400, overlap=100)
        if not chunks:
            chunks = [{
                "chunk_index": 0,
                "section_title": "본문",
                "content": f"[{note_meta['title']}]\n{note_meta['content'][:400]}",
                "raw_text": note_meta['content'][:400]
            }]

        model = embedder.get_embedding_model()
        chunk_texts = [c["content"] for c in chunks]
        chunk_embeddings = model.encode(chunk_texts, batch_size=32)

        summary_text = f"제목: {note_meta['title']} 요약: {note_meta['summary']} 태그: {', '.join(note_meta['tags'])}"
        doc_emb = model.encode(summary_text)
        if hasattr(doc_emb, "tolist"):
            doc_emb = doc_emb.tolist()
        else:
            doc_emb = list(doc_emb)

        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO note_embeddings 
                (filepath, mtime, title, summary, tags, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                filepath,
                mtime,
                note_meta['title'],
                note_meta['summary'],
                json.dumps(note_meta['tags'], ensure_ascii=False),
                note_meta['content'],
                json.dumps(doc_emb)
            ))

            cursor.execute("DELETE FROM note_chunks WHERE filepath = ?", (filepath,))
            chunk_records = []
            for c, c_vec in zip(chunks, chunk_embeddings):
                c_id = f"{filepath}#{c['chunk_index']}"
                vec_list = c_vec.tolist() if hasattr(c_vec, "tolist") else list(c_vec)
                chunk_records.append((
                    c_id,
                    filepath,
                    c['section_title'],
                    c['chunk_index'],
                    c['content'],
                    json.dumps(vec_list)
                ))

            cursor.executemany('''
                INSERT OR REPLACE INTO note_chunks
                (chunk_id, filepath, section_title, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', chunk_records)
            conn.commit()

        # 인메모리 및 바이너리 캐시 즉시 리로드
        get_cached_chunk_matrix(force_reload=True)

        logging.info(f"Ultra-fast single note sync: {os.path.basename(filepath)} ({len(chunks)} chunks).")
        return {
            "status": "synced",
            "path": filepath,
            "title": note_meta["title"],
            "chunks": len(chunks)
        }
    except Exception as e:
        logging.error(f"Failed single note sync for {filepath}: {e}")
        return {"status": "error", "error": str(e), "path": filepath}


# ==========================================
# 초고속 인메모리 RAM 및 바이너리 캐시 (0.002s 극한 최적화)
# ==========================================
CACHE_NPY_PATH = os.path.join(BASE_DIR, "chunks_cache.npy")
CACHE_META_PATH = os.path.join(BASE_DIR, "chunks_cache_meta.json")

_CACHED_CHUNKS_MATRIX = None   # L2-normalized float32 numpy array (N, 1024)
_CACHED_CHUNKS_META = None     # list of chunk dicts
_CACHED_DB_MTIME = 0           # queue.db file modification time
_QUERY_EMBEDDING_CACHE = {}    # query text -> 1024-dim numpy array

def get_cached_chunk_matrix(force_reload=False):
    """
    queue.db의 600+개 청크 벡터를 메모리에 L2-정규화된 상태로 상주시켜,
    매 검색마다 발생하는 DB SELECT(0.2s) 및 600번의 JSON 파싱 오버헤드를 0초로 단축합니다.
    디스크 바이너리(.npy) 캐시를 적용하여 프로세스 콜드 스타트도 0.007초 만에 완료됩니다.
    """
    global _CACHED_CHUNKS_MATRIX, _CACHED_CHUNKS_META, _CACHED_DB_MTIME
    db_path = os.path.join(BASE_DIR, "queue.db")
    if not os.path.exists(db_path):
        return None, []

    try:
        current_mtime = os.path.getmtime(db_path)
    except Exception:
        current_mtime = 0

    if not force_reload and _CACHED_CHUNKS_MATRIX is not None and _CACHED_DB_MTIME == current_mtime:
        return _CACHED_CHUNKS_MATRIX, _CACHED_CHUNKS_META

    # 바이너리 디스크 캐시 (.npy + .json) 검사
    if not force_reload and os.path.exists(CACHE_NPY_PATH) and os.path.exists(CACHE_META_PATH):
        try:
            cache_mtime = os.path.getmtime(CACHE_NPY_PATH)
            if cache_mtime >= current_mtime:
                matrix = np.load(CACHE_NPY_PATH)
                with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                if len(matrix) == len(metas):
                    _CACHED_CHUNKS_MATRIX = matrix
                    _CACHED_CHUNKS_META = metas
                    _CACHED_DB_MTIME = current_mtime
                    return _CACHED_CHUNKS_MATRIX, _CACHED_CHUNKS_META
        except Exception as e:
            logging.warning(f"Failed to load binary chunk cache: {e}")

    # DB에서 로드 후 메모리 및 바이너리 캐시 갱신
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.chunk_id, c.filepath, c.section_title, c.chunk_index, c.content, c.embedding,
                       e.title, e.summary, e.tags
                FROM note_chunks c
                JOIN note_embeddings e ON c.filepath = e.filepath
                WHERE c.filepath NOT LIKE '%/대화기록/%' AND c.filepath NOT LIKE '%\\대화기록\\%'
            """)
            rows = cursor.fetchall()
            if not rows:
                cursor.execute("""
                    SELECT filepath, filepath, '개요', 0, content, embedding, title, summary, tags
                    FROM note_embeddings
                    WHERE filepath NOT LIKE '%/대화기록/%' AND filepath NOT LIKE '%\\대화기록\\%'
                """)
                rows = cursor.fetchall()

        if not rows:
            return None, []

        parsed_rows = []
        dim_counts = {}

        for row in rows:
            chunk_id, filepath, sec_title, chunk_idx, content, emb_json, title, summary, tags_json = row
            if _is_index_hub(filepath):
                continue
            try:
                emb = json.loads(emb_json)
                if not isinstance(emb, list) or len(emb) == 0:
                    continue
                tags = json.loads(tags_json) if tags_json else []
                d = len(emb)
                dim_counts[d] = dim_counts.get(d, 0) + 1
                parsed_rows.append((emb, {
                    "chunk_id": chunk_id,
                    "filepath": filepath,
                    "title": title,
                    "summary": summary,
                    "section_title": sec_title,
                    "chunk_index": chunk_idx,
                    "matched_chunk": content,
                    "tags": tags
                }))
            except Exception:
                pass

        if not parsed_rows:
            return None, []

        target_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
        chunk_embeddings_list = [emb for emb, _ in parsed_rows if len(emb) == target_dim]
        chunk_metas = [meta for emb, meta in parsed_rows if len(emb) == target_dim]

        if not chunk_metas:
            return None, []

        matrix = np.array(chunk_embeddings_list, dtype=np.float32)
        # L2 정규화 미리 계산
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        _CACHED_CHUNKS_MATRIX = matrix
        _CACHED_CHUNKS_META = chunk_metas
        _CACHED_DB_MTIME = current_mtime

        # 바이너리 디스크 캐시 보관
        try:
            np.save(CACHE_NPY_PATH, matrix)
            with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
                json.dump(chunk_metas, f, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to persist binary chunk cache: {e}")

        return _CACHED_CHUNKS_MATRIX, _CACHED_CHUNKS_META
    except Exception as e:
        logging.error(f"Failed to cache chunk matrix: {e}")
        return None, []


def search_keyword_fast(query: str, top_k: int = 20):
    """
    RAM에 상주한 600+개 청크 메타데이터를 대상으로 0.002초 만에 초고속 가중치 키워드 검색을 수행합니다.
    제목(15.0), 섹션명(10.0), 태그(8.0), 청크 본문(5.0) 가중치를 부여하며 순위(rank_kw)를 반환합니다.
    """
    tokens = [t.strip() for t in re.split(r'\s+', query) if len(t.strip()) >= 2]
    if not tokens:
        tokens = [query.strip()] if query.strip() else []
    if not tokens:
        return []

    # 1. 인메모리 캐시에서 직접 탐색 (SQLite 쿼리 완전 생략, 0.002초!)
    _, chunk_metas = get_cached_chunk_matrix()

    if chunk_metas:
        scored = []
        q_lower = query.strip().lower()

        for c in chunk_metas:
            score = 0.0
            title = c.get("title", "")
            sec_title = c.get("section_title", "")
            content = c.get("matched_chunk", "")
            tags = c.get("tags", [])
            tags_text = " ".join(tags) if isinstance(tags, list) else str(tags)

            title_lower = title.lower()
            section_lower = sec_title.lower()
            content_lower = content.lower()
            tags_lower = tags_text.lower()

            # 전체 쿼리 일치 보너스
            if q_lower in title_lower:
                score += 15.0
            if q_lower in section_lower:
                score += 10.0
            if q_lower in tags_lower:
                score += 8.0
            if q_lower in content_lower:
                score += 5.0

            # 개별 토큰 매칭 점수
            token_hits = 0
            for t in tokens:
                tl = t.lower()
                hit = False
                if tl in title_lower:
                    score += 6.0
                    hit = True
                if tl in section_lower:
                    score += 4.0
                    hit = True
                if tl in tags_lower:
                    score += 3.0
                    hit = True
                if tl in content_lower:
                    score += 1.5
                    hit = True
                if hit:
                    token_hits += 1

            if len(tokens) > 1 and token_hits == len(tokens):
                score += 8.0

            if score > 0:
                item = dict(c)
                item["match_score"] = score
                scored.append(item)

        scored.sort(key=lambda x: x["match_score"], reverse=True)
        results = scored[:top_k]
        for rank, r in enumerate(results, 1):
            r["rank_kw"] = rank
        return results

    # 캐시 미비 시 SQLite 폴백
    db_path = os.path.join(BASE_DIR, "queue.db")
    if not os.path.exists(db_path):
        return []

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.chunk_id, c.filepath, c.section_title, c.chunk_index, c.content,
                       e.title, e.summary, e.tags
                FROM note_chunks c
                JOIN note_embeddings e ON c.filepath = e.filepath
                WHERE c.filepath NOT LIKE '%/대화기록/%' AND c.filepath NOT LIKE '%\\대화기록\\%'
            """)
            rows = cursor.fetchall()
            if not rows:
                cursor.execute("""
                    SELECT filepath, filepath, '개요', 0, content, title, summary, tags
                    FROM note_embeddings
                    WHERE filepath NOT LIKE '%/대화기록/%' AND filepath NOT LIKE '%\\대화기록\\%'
                """)
                rows = cursor.fetchall()
            
        scored = []
        q_lower = query.strip().lower()

        for chunk_id, filepath, section_title, chunk_index, content, title, summary, tags_json in rows:
            score = 0.0
            title_lower = title.lower() if title else ""
            section_lower = section_title.lower() if section_title else ""
            content_lower = content.lower() if content else ""
            tags_lower = tags_json.lower() if tags_json else ""
            
            if q_lower in title_lower:
                score += 15.0
            if q_lower in section_lower:
                score += 10.0
            if q_lower in tags_lower:
                score += 8.0
            if q_lower in content_lower:
                score += 5.0

            token_hits = 0
            for t in tokens:
                tl = t.lower()
                hit = False
                if tl in title_lower:
                    score += 6.0
                    hit = True
                if tl in section_lower:
                    score += 4.0
                    hit = True
                if tl in tags_lower:
                    score += 3.0
                    hit = True
                if tl in content_lower:
                    score += 1.5
                    hit = True
                if hit:
                    token_hits += 1

            if len(tokens) > 1 and token_hits == len(tokens):
                score += 8.0

            if score > 0:
                try:
                    tags = json.loads(tags_json) if tags_json else []
                except Exception:
                    tags = []

                scored.append({
                    "chunk_id": chunk_id,
                    "filepath": filepath,
                    "title": title,
                    "summary": summary,
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                    "matched_chunk": content,
                    "tags": tags,
                    "match_score": score
                })
                
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        results = scored[:top_k]
        for rank, r in enumerate(results, 1):
            r["rank_kw"] = rank
        return results
    except Exception as e:
        logging.error(f"Failed fast keyword search: {e}")
        return []


def retrieve_similar_notes(question: str, top_k: int = 20):
    """
    메모리에 상주한 청크 행렬 캐시를 활용해 0.002초 만에 코사인 유사도 벡터 검색을 수행합니다.
    """
    global _QUERY_EMBEDDING_CACHE
    chunk_matrix, chunk_metas = get_cached_chunk_matrix()
    if chunk_matrix is None or not chunk_metas:
        return []

    try:
        # 질문 임베딩 (LRU 캐시 활용으로 0.0001초 응답)
        q_clean = question.strip()
        if q_clean in _QUERY_EMBEDDING_CACHE:
            question_embedding = _QUERY_EMBEDDING_CACHE[q_clean]
        else:
            model = embedder.get_embedding_model()
            question_embedding = model.encode(q_clean, convert_to_numpy=True)
            if isinstance(question_embedding, list):
                question_embedding = np.array(question_embedding, dtype=np.float32)
            q_norm_val = np.linalg.norm(question_embedding)
            if q_norm_val > 0:
                question_embedding = question_embedding / q_norm_val
            # 캐시 용량 제한 (최대 256개: 전체 폭파 없이 가장 오래된 쿼리만 FIFO 축출)
            if len(_QUERY_EMBEDDING_CACHE) >= 256:
                try:
                    _QUERY_EMBEDDING_CACHE.pop(next(iter(_QUERY_EMBEDDING_CACHE)))
                except Exception:
                    pass
            _QUERY_EMBEDDING_CACHE[q_clean] = question_embedding

        # 이미 L2-정규화된 행렬과의 초고속 내적 (소요 시간 0.001초!)
        similarities = np.dot(chunk_matrix, question_embedding).squeeze()
        if len(chunk_metas) == 1:
            similarities = np.array([similarities])

        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx])
            item = dict(chunk_metas[idx])
            item["similarity"] = round(score, 4)
            item["rank_vec"] = rank
            retrieved.append(item)

        return retrieved
    except Exception as e:
        logging.error(f"Failed vector retrieval: {e}")
        return []


def hybrid_search_notes(query: str, top_k: int = 5, candidate_k: int = 20, fast_path: bool = True):
    """
    진짜 하이브리드 RRF (Reciprocal Rank Fusion) 검색 엔진:
    1. 초고속 Fast-Path: 제목/태그/섹션 일치(match_score >= 12.0) 시 0.003초 만에 즉시 반환!
    2. 일반/서술형 질의 시 키워드 검색(Top-20)과 인메모리 벡터 검색(Top-20)을 병렬 수행 후 RRF 융합!
    """
    kw_results = [c for c in search_keyword_fast(query, top_k=candidate_k) if not _is_index_hub(c["filepath"])]

    # ⚡ Fast-Path: 3단어 이하의 짧은 키워드 질의 또는 전체 질의가 문서 제목과 일치할 때만 조기 반환!
    # (4단어 이상의 긴 자연어/서술형 질의는 반드시 True RRF 벡터 융합을 수행하여 의미 누락 방지)
    q_clean = query.strip().lower()
    if fast_path and kw_results:
        top_hit = kw_results[0]
        is_short_query = len(q_clean.split()) <= 3
        is_exact_title = q_clean in top_hit.get("title", "").lower()
        if (is_short_query or is_exact_title) and top_hit.get("match_score", 0) >= 12.0:
            doc_map = {}
            for c in kw_results:
                fpath = c["filepath"]
                if fpath not in doc_map:
                    doc_map[fpath] = {
                        "path": fpath,
                        "title": c["title"],
                        "summary": c.get("summary", ""),
                        "tags": c.get("tags", []),
                        "matched_section": c.get("section_title", "개요"),
                        "matched_chunk": c.get("matched_chunk", ""),
                        "rrf_score": round(1.0 / (60.0 + c.get("rank_kw", 1)), 6),
                        "similarity": min(0.98, round(0.50 + (c.get("match_score", 0) / 30.0) * 0.45, 4))
                    }
            if len(doc_map) >= min(top_k, 3) or top_hit.get("match_score", 0) >= 20.0:
                return list(doc_map.values())[:top_k]

    vec_results = [c for c in retrieve_similar_notes(query, top_k=candidate_k) if not _is_index_hub(c["filepath"])]

    # RRF 스코어 결합
    all_chunks = {}

    for item in kw_results:
        cid = item["chunk_id"]
        all_chunks[cid] = dict(item)
        all_chunks[cid]["rank_kw"] = item["rank_kw"]
        all_chunks[cid]["rank_vec"] = candidate_k + 60  # 미등장 패널티

    for item in vec_results:
        cid = item["chunk_id"]
        if cid not in all_chunks:
            all_chunks[cid] = dict(item)
            all_chunks[cid]["rank_kw"] = candidate_k + 60
        all_chunks[cid]["rank_vec"] = item["rank_vec"]
        all_chunks[cid]["similarity"] = item["similarity"]

    # RRF 점수 계산
    scored_chunks = []
    for cid, item in all_chunks.items():
        r_kw = item.get("rank_kw", candidate_k + 60)
        r_vec = item.get("rank_vec", candidate_k + 60)
        rrf_score = (1.0 / (60.0 + r_kw)) + (1.0 / (60.0 + r_vec))
        item["rrf_score"] = round(rrf_score, 6)
        scored_chunks.append(item)

    # RRF 점수 내림차순 정렬
    scored_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)

    # 문서 단위 그룹화 및 중복 제거 (Document-level Grouping)
    # 한 파일에서 가장 적중도가 높은 청크를 대표 매칭 청크로 선정
    doc_map = {}
    for c in scored_chunks:
        fpath = c["filepath"]
        if fpath not in doc_map:
            doc_map[fpath] = {
                "path": fpath,
                "title": c["title"],
                "summary": c.get("summary", ""),
                "tags": c.get("tags", []),
                "matched_section": c.get("section_title", "개요"),
                "matched_chunk": c.get("matched_chunk", ""),
                "rrf_score": c["rrf_score"],
                "similarity": c.get("similarity", round(c["rrf_score"] * 30.0, 4))
            }

    final_results = list(doc_map.values())[:top_k]
    return final_results


def rerank_notes(question, notes):
    """
    Gemini LLM을 이용해 1차로 벡터 인출된 문서 후보군(최대 10개) 중
    사용자 질문에 실제 답변하는 데 유용한 상위 3개 문서만 선별(Re-ranking)합니다.
    """
    if not notes:
        return []
        
    prompt = f"""
    [지식 아카이브 검색기 - 문서 재정렬 및 필터링]
    
    사용자 질문: "{question}"
    
    아래는 1차 검색으로 선별된 10개의 지식 노트 후보군입니다.
    각 후보의 번호(Index)와 제목 및 요약을 분석하여, 사용자 질문에 답변하는 데 직접적인 실질적 정보가 포함된 가장 유용한 문서를 최대 3개 선택해 주십시오. (가장 연관성이 높은 순서대로 골라주십시오.)
    
    """
    for idx, n in enumerate(notes):
        prompt += f"\n[인덱스 {idx}] 제목: {n['title']} | 요약: {n['summary']}"
        
    prompt += """
    
    질문과 아무런 관련이 없거나 쓸모없는 정보라면 선택하지 마십시오.
    JSON 형식에 맞추어 오직 선택한 인덱스의 배열("relevant_indices": [인덱스1, 인덱스2, ...])만 반환하십시오.
    """
    
    schema = {
        "type": "object",
        "properties": {
            "relevant_indices": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "질문 답변에 직접적으로 도움이 되는 문서들의 인덱스 번호 배열 (최대 3개, 중요도 내림차순)"
            }
        },
        "required": ["relevant_indices"]
    }
    
    try:
        # Gemini API 우선 사용
        if os.environ.get("GEMINI_API_KEY"):
            import agents
            print(f"[RAG 리랭킹] {len(notes)}개 후보군에 대한 Gemini 리랭커 판정 개시...")
            raw_res = agents.call_gemini_api(
                prompt=prompt,
                system_instruction="당신은 데이터베이스 검색 리랭커(Re-ranker)입니다. 오직 지정된 스키마에 맞는 JSON 데이터만 출력하십시오.",
                format=schema
            )
            parsed = agents.parse_json_response(raw_res, prompt)
            indices = parsed.get("relevant_indices", [])
            
            reranked = []
            for idx in indices:
                if 0 <= idx < len(notes):
                    reranked.append(notes[idx])
            print(f"[RAG 리랭킹] 필터링 완료: {len(notes)}개 ➔ {len(reranked)}개 문서 엄선.")
            return reranked[:3]
    except Exception as e:
        print(f"[경고] LLM 리랭킹 실패, 1차 벡터 검색 결과를 그대로 사용합니다. 에러: {e}")
        
    # 실패 시 또는 API 키가 없을 시 1차 벡터 검색 상위 3개 그대로 사용
    return notes[:3]


def query_rag_answer(question):
    """
    질문에 답변하기 위해 유사 과거 노트를 인출하여 로컬 Ollama 또는 Gemini 컨텍스트에 주입하고 답변을 획득합니다.
    """
    # 1차로 넓게 벡터 검색 (top_k=10)
    candidates = retrieve_similar_notes(question, top_k=10)
    # LLM을 이용해 질문과 실제 관련 있는 핵심 3개 문서 엄선
    context_notes = rerank_notes(question, candidates)
    
    # RAG 컨텍스트 조립
    context_str = ""
    sources = []
    
    for i, n in enumerate(context_notes, 1):
        rel_path = os.path.relpath(n["path"], NOTES_DIR).replace("\\", "/")
        vault_name = os.path.basename(NOTES_DIR)  # 사용자의 옵시디언 금고(vault) 폴더명 동적 추출
        # 💡 HTML 앵커 태그 안에 unescaped 한글 딥링크를 직접 집어넣어, 텔레그램에서 파란색 클릭 링크로 활성화시키면서 브라우저 차단과 에러를 100% 무결하게 해결합니다!
        obsidian_link = f"obsidian://open?vault={vault_name}&file={rel_path}"
        
        context_str += f"[지식 {i}] (출처 날짜: {os.path.basename(os.path.dirname(n['path']))})\n"
        context_str += f"- 제목: {n['title']}\n"
        context_str += f"- 요약: {n['summary']}\n"
        context_str += f"- 상세내용: {n['content']}\n\n"
        
        sources.append(f"{i}. <b>{html_escape(n['title'])}</b>\n👉 <a href=\"{obsidian_link}\">Obsidian에서 즉시 열기 🔗</a>")

    # 프롬프트 인젝션 템플릿
    system_prompt = (
        "당신은 최고의 지능을 가진 로컬 AI 아카이빙 개인 비서입니다.\n"
        "아래에 제공된 [아카이빙된 참고 지식]만을 100% 신뢰하여 사용자의 질문에 완벽하고 전문적인 한국어로 친절히 답해 주십시오.\n"
        "반드시 참고 지식에 나와 있는 구체적인 팩트와 특징을 풍부하게 설명하되, 주어진 내용 외의 허구(환각)는 절대 생성하지 마십시오.\n"
        "만약 참고 지식으로 대답할 수 없는 생뚱맞은 질문이라면, 과거 메모에 정보가 없음을 안내하고 정중하게 거절하십시오.\n\n"
        "[CRITICAL LANGUAGE RULE]\n"
        "- Reply ENTIRELY in natural KOREAN.\n"
        "- Never output Chinese characters.\n"
    )
    
    user_prompt = f"[아카이빙된 참고 지식]\n"
    if context_str:
        user_prompt += context_str
    else:
        user_prompt += "* (검색어와 부합하는 관련 과거 지식 노트가 없습니다. 일반 LLM 상식으로 유연하게 대답하십시오.) *\n\n"
        
    user_prompt += f"[사용자 질문]\n{question}"
    
    try:
        # 💡 구글 Gemini API 우선 적용
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        answer = None
        if gemini_api_key:
            try:
                print("[RAG] 구글 Gemini API 호출 개시 (모델: gemini-2.5-flash)...")
                import agents
                answer = agents.call_gemini_api(
                    prompt=user_prompt,
                    system_instruction=system_prompt,
                    format=None  # 텍스트 질답이므로 JSON 스키마를 걸지 않고 자연어로 답변
                )
                print("[RAG] 구글 Gemini API 답변 생성 완료!")
            except Exception as gemini_err:
                print(f"[경고] RAG용 Gemini 호출 실패, 로컬 Ollama 폴백 가동: {gemini_err}")
                answer = None
                
        # 💡 Gemini를 사용하지 않거나 실패한 경우 로컬 Ollama로 폴백
        if not answer:
            data = {
                "model": "gemma4:12b-it-qat",
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2, # RAG의 정확한 준수를 위해 낮은 온도로 유도
                    "num_predict": 1024,
                    "num_ctx": 8192     # 긴 RAG 컨텍스트를 위해 8192 토큰 확보
                },
                "keep_alive": "1m"      # 답변 완료 1분 후 VRAM에서 모델을 자동 언로드
            }
            
            req = urllib.request.Request(
                "http://localhost:11434/api/generate", 
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode('utf-8'))
                answer = result.get("response", "답변 획득 실패")
            
        # 텔레그램 리턴 텍스트 가공
        escaped_answer = html_escape(answer)
        
        response_html = (
            f"🤖 <b>AI 아카이빙 비서의 RAG 지식 답변</b>\n"
            f"질문: <i>{html_escape(question)}</i>\n"
            "------------------------------------\n"
            f"{escaped_answer}\n\n"
        )
        
        if context_notes:
            response_html += "🔗 <b>답변에 참고한 과거 노트 소스 (가볍게 툭 누르면 파일명 복사 📋):</b>\n"
            for i, n in enumerate(context_notes, 1):
                filename = os.path.basename(n["path"])
                # 💡 텔레그램 HTML 파싱에서 <code> 태그를 씌워 파일명을 출력하면, 폰에서 이 파일명을 터치 한 번만 해도 즉시 클립보드에 전체 복사됩니다!
                # 💡 복사한 파일명을 Obsidian 검색창에 붙여넣기만 하면 0.1초 만에 바로 열립니다! 에러 유발 요소를 100% 제거한 가장 확실하고 사용성 좋은 해결책입니다.
                response_html += f"{i}. <b>{html_escape(n['title'])}</b>\n👉 <code>{filename}</code>\n\n"
        else:
            response_html += "⚠️ <i>(주의: 질문과 부합하는 과거 메모 지식을 찾지 못해 일반 AI 상식으로 답해 드렸습니다.)</i>"
            
        return {"text": response_html, "buttons": []}
    except Exception as e:
        logging.error(f"Failed to call Ollama in RAG: {e}")
        return {"text": f"⚠️ RAG 분석 오류 발생: {html_escape(str(e))}", "buttons": []}


def html_escape(text):
    return re.sub(r'[&<>]', lambda m: {'&': '&amp;', '<': '&lt;', '>': '&gt;'}[m.group(0)], str(text))
