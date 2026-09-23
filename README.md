# RAG Lab — 검색증강생성(RAG) 시스템 구축

문서 파싱부터 평가까지, **브라우저에서 직접 실습하며** 동작하는 RAG를 만드는 12주 웹 강좌입니다.
같은 콘텐츠를 **학생용(문서 + 실습)**과 **교사용(PPT 슬라이드 + 교사 노트)** 두 화면으로 제공합니다.

- 🌐 사이트: https://samcho93.github.io/studyRAG/
- 🎓 학생용: https://samcho93.github.io/studyRAG/student.html
- 🧑‍🏫 교사용: https://samcho93.github.io/studyRAG/teacher.html

## 화면 구성

| 왼쪽 | 가운데 | 오른쪽 |
|---|---|---|
| 주차 목차 · 현재 주차의 섹션(학생용) / 슬라이드 목록(교사용) | 🎓 문서(학습 목표 · 왜 필요한가 · 개념 · 실습 위젯 · Challenge · 정리)<br>🧑‍🏫 16:9 슬라이드 + 교사 노트, 끝에 부록(강의안 · 정답 · 막히는 지점 · 토론 · 루브릭) | **실습 결과** — 위젯 출력, Challenge 코드 ▶ 실행 결과(console.log · console.table) |

- 1000px 미만(휴대폰)에서는 목록이 서랍 메뉴로 바뀌고, 실습 결과는 위젯·코드 바로 아래에 표시됩니다.
- Challenge 코드는 페이지 안에서 바로 고치고 실행합니다. Web Worker에서 돌아가므로 무한 루프도 10초 뒤 자동으로 멈춥니다.
- 임베딩은 [Transformers.js](https://huggingface.co/docs/transformers.js)로 브라우저 안에서 실행합니다(5주차부터, 모델 최초 1회 다운로드).
- LLM이 필요한 주차는 학생 본인의 API 키를 쓰며, 키는 `sessionStorage`에만 저장됩니다.

교사용 슬라이드 단축키: `←` `→` 이동 · `F` 전체 화면 · `N` 교사 노트 · `B` 화면 가리기 · `T` 타이머 · `P` 발표자 창

## 커리큘럼 (12주)

| 주차 | 주제 | 실습 위젯 | 상태 |
|---|---|---|---|
| 01 | LLM 기초 & RAG 필요성 | 토큰 카운터, 환각 체험 | 준비 중 |
| 02 | 프롬프트 엔지니어링 | 프롬프트 A/B 비교 | 준비 중 |
| 03 | 문서 로딩 & 파싱 | PDF 텍스트 추출 뷰어 | 준비 중 |
| 04 | 청킹 전략 | **청킹 플레이그라운드** | ✅ 공개 |
| 05 | 임베딩 & 유사도 | 유사도 히트맵 + 2D 투영 | 준비 중 |
| 06 | 벡터DB & 인덱스 | Flat vs HNSW 속도 비교 | 준비 중 |
| 07 | 최소 RAG 완성 | 전체 파이프라인 실행 | 준비 중 |
| 08 | 하이브리드 검색 + 리랭킹 | BM25 vs 벡터 vs 하이브리드 | 준비 중 |
| 09 | 평가 데이터셋 & 측정 | 점수판(Recall@k, nDCG) | 준비 중 |
| 10 | 프레임워크 리팩터링 | 코드 비교 뷰어 | 준비 중 |
| 11 | 웹 UI + 인용 표기 | 인용 하이라이트 데모 | 준비 중 |
| 12 | 팀 프로젝트 발표 | 팀별 결과 업로드 | 준비 중 |

## 폴더 구조

```
index.html · student.html · teacher.html   랜딩 / 학생용·교사용 진입점
weeks/wNN/index.html · teacher.html         주차별 학생용 문서 / 교사용 슬라이드
teacher/index.html                          교사용 허브
assets/css/                                 tokens(디자인 토큰) · base · components(3단 셸) · slides
assets/js/core/                             chunker · embed · bm25 · vectorstore · rerank · metrics · llm
assets/js/widgets/                          주차별 실습 위젯 (위젯 하나 = 파일 하나)
assets/js/site/                             목차·진도·테마 · 슬라이드 엔진 · 실습 결과 창 · 코드 실행기
assets/data/corpus/ · golden/               전 과정 공통 문서셋 / 평가용 골든셋
CLAUDE.md                                   작업 규칙
```

## 로컬에서 보기

ES Module과 fetch는 `file://`에서 동작하지 않으므로 로컬 서버로 엽니다.

```bash
python -m http.server 8000
```

→ http://localhost:8000

## 배포

빌드 과정이 없습니다. `main` 브랜치에 push하면 GitHub Pages(Settings → Pages → Branch `main` / root)가 그대로 배포합니다.
