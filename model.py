from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def get_embedding_model(model_name="intfloat/multilingual-e5-base"):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


def get_reranker_model(model_name="BAAI/bge-reranker-v2-m3"):
    return HuggingFaceCrossEncoder(model_name=model_name)

def get_default_profile():
    return {
    "user_id": "abc123",
    "portfolio": ["삼성전자", "네이버", "두산로보틱스"],
    "interests": ["AI 반도체", "핀테크", "로봇"],
    "query_history": ["네이버 실적", "삼성전자 하락 이유"],
    "alert_schedule": "매일 18:00",
    "user_preference": """[제목]: 보고서 제목
  [핵심 요약]:\n
- 이 세부 조사항목에 대한 요약된 결론을 한두 문장으로 정리하세요.

[배경 및 주요 사실]:\n
- 관련된 배경 정보, 수치, 시기 등을 요약하세요.

[세부 분석]:\n
- 원인, 영향, 관점 차이 등 심화 내용을 서술하세요.
- 다양한 관점이 존재한다면 관점의 출처와 함께 기술하세요.

[향후 전망 또는 시사점]:\n
- 이 이슈가 앞으로 어떤 영향을 미칠 수 있는지 서술하세요.
- 관련 기업이나 산업에 대한 함의도 포함할 수 있습니다.""",
}