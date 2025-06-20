# web_rag_economy_trial

## 프로젝트 개요

이 프로젝트는 **웹을 데이터베이스처럼 활용하여 정보를 검색하고**, 그 결과를 LLM의 context로 통합하는 **Web-RAG 구조 구현**을 목표로 합니다.

- **프로젝트 기간**: 2일
- **목적**: 최신 뉴스 정보에 기반한 자동 경제 보고서 생성

📎 자세한 구현 및 예시 결과는 [`report.pdf`](report.pdf)를 참고하세요.

---

## 🔍 주요 전략

- **Web Search Strategy**
  - Google News RSS 기반의 URL 검색 전략 및 파싱
- **Query Expansion**
  - 유저 요청에 맞는 검색 키워드 생성 및 reasoning 범위 확장
- **Retrieval**
  - 중복 제거 및 정보 다양성 확보를 위한 MMR
  - 최신성 보장을 위한 timestamp prefix 추가 rerank 전략

## 테스트 방법

아래 명령어로 `main_llm.py`를 실행하고, 안내에 따라 OpenAI API 키와 질의(`user_query`)를 입력하면 자동으로 웹 기반 정보를 검색하여 보고서를 생성합니다.

```bash
python main_llm.py
```

## 프로젝트 구조

```text
.
├── google_parsing.py     # Google News RSS 검색조건 및 파싱 로직
├── main_llm.py           # 메인 실행 파일 (API Key, 질의 입력 → 보고서 생성)
├── model.py              # embedding, reranker 모델 로딩 함수
├── retrieval.py          # 문서 유사도 기반 검색, MMR, rerank 처리
├── util.py               # 유틸리티 함수, chunk DB 관리 및 웹서칭 필요여부 판단 로직
├── report.pdf            # 발표자료 (출력 결과 예시 포함)