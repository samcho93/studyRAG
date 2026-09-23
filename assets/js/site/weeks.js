// Single source of truth for the 12-week curriculum.
// `ready: true` marks weeks whose pages exist; update it when a week ships.

export const WEEKS = [
  { no: 1, title: 'LLM 기초 & RAG 필요성', widget: '토큰 카운터, 환각 체험', llm: '선택', ready: false },
  { no: 2, title: '프롬프트 엔지니어링', widget: '프롬프트 A/B 비교', llm: '필요', ready: false },
  { no: 3, title: '문서 로딩 & 파싱', widget: 'PDF 텍스트 추출 뷰어', llm: '—', ready: false },
  { no: 4, title: '청킹 전략', widget: '청킹 플레이그라운드', llm: '—', ready: true, core: true },
  { no: 5, title: '임베딩 & 유사도', widget: '유사도 히트맵 + 2D 투영', llm: '—', ready: false, core: true },
  { no: 6, title: '벡터DB & 인덱스', widget: 'Flat vs HNSW 속도 비교', llm: '—', ready: false },
  { no: 7, title: '최소 RAG 완성', widget: '전체 파이프라인 실행', llm: '필요', ready: false },
  { no: 8, title: '하이브리드 검색 + 리랭킹', widget: 'BM25 vs 벡터 vs 하이브리드', llm: '—', ready: false, core: true },
  { no: 9, title: '평가 데이터셋 & 측정', widget: '점수판(Recall@k, nDCG)', llm: '선택', ready: false, core: true },
  { no: 10, title: '프레임워크 리팩터링', widget: '코드 비교 뷰어', llm: '—', ready: false },
  { no: 11, title: '웹 UI + 인용 표기', widget: '인용 하이라이트 데모', llm: '필요', ready: false },
  { no: 12, title: '팀 프로젝트 발표', widget: '팀별 결과 업로드', llm: '—', ready: false },
];

export const weekId = (no) => `w${String(no).padStart(2, '0')}`;
