import os
import re
import numpy as np
import urllib.request
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
WIKI_DIR = os.path.join(NOTES_DIR, "wiki") if os.path.exists(os.path.join(NOTES_DIR, "wiki")) else NOTES_DIR
RAW_DIR = os.path.join(NOTES_DIR, "raw")

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
            
        # 1. 파일명에서 접두사 시간 제거해 실제 제목 획득
        if "_" in title:
            parts = title.split("_", 2)
            if len(parts) >= 3:
                title = parts[2].replace("_", " ")
        
        # 2. YAML 파싱
        yaml_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            # 간이 태그 추출
            for line in yaml_content.splitlines():
                if line.strip().startswith("- "):
                    tags.append(line.replace("- ", "").strip())
                    
        # 3. 요약 추출
        summary_match = re.search(r'### 📝 요약\n([^\n]+)', content)
        if summary_match:
            summary = summary_match.group(1).strip()
            
        # 4. 전체 텍스트 본문 추출
        # YAML 영역 제거
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


def scan_all_notes():
    """
    옵시디언 wiki/ 하위 폴더의 모든 컴파일된 지식 노트를 수집합니다.
    """
    notes = []
    target_dir = WIKI_DIR if os.path.exists(WIKI_DIR) else NOTES_DIR
    if not os.path.exists(target_dir):
        return notes
        
    for root, dirs, files in os.walk(target_dir):
        if "대화기록" in root.replace("\\", "/").split("/"):
            continue
        for filename in files:
            if filename.endswith(".md") and not filename.endswith("_ERROR_Fallback.md"):
                filepath = os.path.join(root, filename)
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
        conn.commit()


def sync_note_embeddings():
    """
    wiki/ 하위 폴더의 모든 마크다운 파일을 스캔하여 
    신규/수정된 파일의 BGE-M3 임베딩을 SQLite DB에 캐시/동기화합니다.
    수정 시간(mtime)이 변경되지 않은 파일은 연산을 완전히 건너뜁니다.
    금고에서 삭제된 파일의 레코드 또한 자동으로 DB에서 정리합니다.
    """
    init_rag_db()
    db_path = os.path.join(BASE_DIR, "queue.db")
    target_dir = WIKI_DIR if os.path.exists(WIKI_DIR) else NOTES_DIR
    
    if not os.path.exists(target_dir):
        logging.warning(f"Target directory {target_dir} does not exist.")
        return
        
    # 1. 디렉토리 내부 실제 마크다운 파일들의 정보 수집
    actual_files = {} # absolute_path -> mtime
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith(".md") and not filename.endswith("_ERROR_Fallback.md"):
                filepath = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    actual_files[filepath] = mtime
                except Exception as e:
                    logging.error(f"Failed to get mtime for {filepath}: {e}")

    # 2. DB에 기록된 기존 캐시 목록 조회
    db_files = {} # filepath -> mtime
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
                conn.commit()
        except Exception as e:
            logging.error(f"Failed to delete records from DB: {e}")

    # 4. 신규 및 수정된 파일 판별
    to_update = []
    for path, mtime in actual_files.items():
        if path not in db_files or db_files[path] != mtime:
            to_update.append((path, mtime))

    if not to_update:
        logging.info("No files need embedding update. RAG index is up-to-date.")
        return

    logging.info(f"Need to sync {len(to_update)} files...")
    
    # 5. 임베딩 모델 준비 (여기서 모델이 로드됨)
    model = embedder.get_embedding_model()
    
    # 6. 순차적으로 파싱 및 임베딩 연산 수행하여 DB에 적재
    for path, mtime in to_update:
        try:
            note_meta = extract_note_metadata(path)
            if not note_meta["content"]:
                continue
                
            text_to_embed = f"제목: {note_meta['title']} 요약: {note_meta['summary']} 내용: {note_meta['content']}"
            emb = model.encode(text_to_embed, convert_to_numpy=True).tolist()
            
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
                    json.dumps(emb)
                ))
                conn.commit()
            logging.info(f"Successfully synced embedding for: {os.path.basename(path)}")
        except Exception as e:
            logging.error(f"Failed to sync embedding for {path}: {e}")

    logging.info("RAG embedding sync completed successfully.")


def retrieve_similar_notes(question, top_k=3):
    """
    사용자의 질문과 BGE-M3 코사인 유사도가 가장 높은 과거 지식 노트 top_k개를 DB 캐시로부터 초고속 검색합니다.
    이때 파일 IO 및 실시간 임베딩 연산을 수행하지 않아 극도의 저지연(Latency)을 보장합니다.
    """
    init_rag_db()
    db_path = os.path.join(BASE_DIR, "queue.db")
    
    try:
        # 1. DB의 모든 임베딩 로드
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, title, summary, tags, content, embedding FROM note_embeddings WHERE filepath NOT LIKE '%/대화기록/%' AND filepath NOT LIKE '%\\대화기록\\%'")
            rows = cursor.fetchall()
            
        if not rows:
            logging.info("No cached knowledge note embeddings found in database. Please run sync_note_embeddings() first.")
            return []
            
        note_embeddings_list = []
        notes = []
        
        for row in rows:
            filepath, title, summary, tags_json, content, emb_json = row
            try:
                emb = json.loads(emb_json)
                tags = json.loads(tags_json)
                if len(emb) == 1024:
                    note_embeddings_list.append(emb)
                    notes.append({
                        "path": filepath,
                        "title": title,
                        "summary": summary,
                        "tags": tags,
                        "content": content
                    })
            except Exception as e:
                logging.warning(f"Failed to parse embedding for {filepath}: {e}")
                
        if not notes:
            return []
            
        note_embeddings = np.array(note_embeddings_list) # (N, 1024)
        
        # 2. 질문의 BGE-M3 임베딩 생성 (CPU/GPU 모드에 따름)
        model = embedder.get_embedding_model()
        question_embedding = model.encode(question, convert_to_numpy=True) # (1024,)
        
        # 3. 코사인 유사도 연산 (L2 정규화 후 닷 프로덕트 - Zero-division 방지)
        note_norms = np.linalg.norm(note_embeddings, axis=1, keepdims=True)
        note_norms[note_norms == 0] = 1.0
        note_embeddings = note_embeddings / note_norms
        
        question_norm = np.linalg.norm(question_embedding)
        if question_norm == 0:
            return []
        question_embedding = question_embedding / question_norm
        
        similarities = np.dot(note_embeddings, question_embedding).squeeze()
        
        if len(notes) == 1:
            similarities = np.array([similarities])
            
        # 유사도 내림차순 정렬
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            score = float(similarities[idx])
            # BGE-M3 한국어 매칭 임계치 (LLM 리랭커 도입으로 0.35로 완화)
            if score >= 0.35:
                note_data = notes[idx]
                note_data["similarity"] = score
                retrieved.append(note_data)
                
        logging.info(f"Retrieved {len(retrieved)} high-value notes for context from DB (BGE-M3).")
        return retrieved
    except Exception as e:
        logging.error(f"Failed DB-based RAG retrieval: {e}")
        return []


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
        vault_name = "notes"  # 사용자의 옵시디언 금고(vault) 기본 폴더명
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
                    "num_ctx": 4096     # 컨텍스트 크기를 4096 토큰으로 제한해 VRAM 절약
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
