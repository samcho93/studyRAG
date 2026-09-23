// Single source of truth for the 12-week curriculum.
// `ready: true` marks weeks whose pages exist; update it when a week ships.

export const PARTS = [
  { no: 1, title: 'LLM과 문서 준비', sub: 'LLM 기초 · 프롬프트 · 파싱', weeks: [1, 2, 3] },
  { no: 2, title: '검색 파이프라인', sub: '청킹 · 임베딩 · 벡터스토어', weeks: [4, 5, 6] },
  { no: 3, title: 'RAG 완성과 고도화', sub: '최소 RAG · 하이브리드 검색 · 평가', weeks: [7, 8, 9] },
  { no: 4, title: '서비스와 프로젝트', sub: '리팩터링 · 웹 UI · 발표', weeks: [10, 11, 12] },
];

export const WEEKS = [
  { no: 1, icon: '🧠', title: 'LLM 기초 & RAG 필요성', widget: '토큰 카운터, 환각 체험', llm: '선택', ready: false },
  { no: 2, icon: '✍️', title: '프롬프트 엔지니어링', widget: '프롬프트 A/B 비교', llm: '필요', ready: false },
  { no: 3, icon: '📄', title: '문서 로딩 & 파싱', widget: 'PDF 텍스트 추출 뷰어', llm: '—', ready: false },
  { no: 4, icon: '✂️', title: '청킹 전략', widget: '청킹 플레이그라운드', llm: '—', ready: true, core: true },
  { no: 5, icon: '🧭', title: '임베딩 & 유사도', widget: '유사도 히트맵 + 2D 투영', llm: '—', ready: false, core: true },
  { no: 6, icon: '🗄️', title: '벡터DB & 인덱스', widget: 'Flat vs HNSW 속도 비교', llm: '—', ready: false },
  { no: 7, icon: '🔗', title: '최소 RAG 완성', widget: '전체 파이프라인 실행', llm: '필요', ready: false },
  { no: 8, icon: '⚖️', title: '하이브리드 검색 + 리랭킹', widget: 'BM25 vs 벡터 vs 하이브리드', llm: '—', ready: false, core: true },
  { no: 9, icon: '📊', title: '평가 데이터셋 & 측정', widget: '점수판(Recall@k, nDCG)', llm: '선택', ready: false, core: true },
  { no: 10, icon: '🧰', title: '프레임워크 리팩터링', widget: '코드 비교 뷰어', llm: '—', ready: false },
  { no: 11, icon: '💬', title: '웹 UI + 인용 표기', widget: '인용 하이라이트 데모', llm: '필요', ready: false },
  { no: 12, icon: '🏁', title: '팀 프로젝트 발표', widget: '팀별 결과 업로드', llm: '—', ready: false },
];

export const weekId = (no) => `w${String(no).padStart(2, '0')}`;
