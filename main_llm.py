import re
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime
from retrieval import chunk_documents, retrieval_web
from google_parsing import search_googlenews
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import numpy as np
import getpass
from model import get_embedding_model, get_reranker_model, get_default_profile


def expand_query(llm, original_question, user_profile):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_profile_prompt = f"\n사용자 포트폴리오: {user_profile['portfolio'][:5]}\n사용자 관심사: {user_profile['interests'][:5]}\n이전 요청기록: {user_profile['query_history'][:5]}"
    messages = [
        SystemMessage(
            content=(
                f"현재 시각은 {now} 입니다."
                "너는 사용자의 질문에 대한 최종 경제 분석 보고서를 생성해야 합니다."
                "사용자의 질문의 의도를 파악하고 세부 조사 항목을 선택하세요."
                "최근, 요즘 등 불명확한 시간 표현보다는 년, 분기같은 구체적인 시간 표현을 사용하세요. "
                "사용자의 포트폴리오, 관심사, 이전 요청기록 중 사용자의 질문에 관련된 부분만 참고하고, 우선적으로 현재 사용자의 질문에 집중하세요. "
                "보고서를 생성하기 위해 사용자의 질문을 명확하고 구체적인 세부 조사 항목으로 확장하세요."
                "사용자의 질문에 대해 서로 구분되는 다른 관점 및 영역에서 조사하는 세부 조사 항목 3개를 선택하세요."
                "웹서칭에 효과적인 적절한 키워드 형태의 세부 조사 항목 3개를 출력하세요."
                "반드시 다음의 출력 형태를 따르세요"
                "1. '세부 조사 항목1 내용'"
                "2. '세부 조사 항목2 내용'"
                "3. '세부 조사 항목3 내용'"
            )
        ),
        HumanMessage(content=f"질문: {original_question}" + user_profile_prompt),
    ]

    response = llm.invoke(messages).content.strip()
    subqueries = re.findall(r"'(.*?)'", response)
    return subqueries


def expand_query2(llm, original_question, user_profile):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_profile_prompt = f"\n사용자 포트폴리오: {user_profile['portfolio'][:5]}\n사용자 관심사: {user_profile['interests'][:5]}\n이전 요청기록: {user_profile['query_history'][:5]}"
    messages = [
        SystemMessage(
            content=(
                f"현재 시각은 {now} 입니다."
                "너는 웹서칭 정보 검색 전문가입니다. 사용자의 질문에 대한 최종 경제 분석 보고서를 위해 정보를 검색해야 합니다."
                "사용자의 질문의 의도를 파악하고 세부 조사 항목을 선택하세요."
                # "최근, 요즘 등 불명확한 시간 표현보다는 년, 분기같은 구체적인 시간 표현을 사용하세요. "
                "최근, 2025년 같은 시간 표현은 키워드에 포함하지 마세요. "
                "사용자의 포트폴리오, 관심사, 이전 요청기록 중 사용자의 질문에 관련된 부분만 참고하고, 우선적으로 현재 사용자의 질문에 집중하세요. "
                "보고서를 생성하기 위해 사용자의 질문을 명확하고 구체적인 세부 조사 항목으로 확장하세요."
                "사용자의 질문에 대해 서로 구분되는 다른 관점 및 영역에서 조사하는 세부 조사 항목 3개를 선택하세요."
                "웹서칭에 효과적인 적절한 단일 키워드 형태의 세부 조사 항목 3개를 출력하세요."
                "반드시 다음의 출력 형태를 따르세요"
                "1. '세부 조사 항목1 내용'"
                "2. '세부 조사 항목2 내용'"
                "3. '세부 조사 항목3 내용'"
            )
        ),
        HumanMessage(content=f"질문: {original_question}" + user_profile_prompt),
    ]

    response = llm.invoke(messages).content.strip()
    subqueries = re.findall(r"'(.*?)'", response)
    return subqueries


def llm_call_single(llm, original_question, sub_query, user_profile, contexts):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged_context = "\n".join(doc.page_content for doc in contexts)

    user_prompt = (
        f"""질문: "{original_question}"\n\n"""
        + f"세부 조사항목 {sub_query}에 대한 정보:\n{merged_context}"
    )
    user_profile_prompt = f"\n사용자 포트폴리오: {user_profile['portfolio'][:5]}\n사용자 관심사: {user_profile['interests'][:5]}\n이전 요청기록: {user_profile['query_history'][:5]}"

    messages = [
        SystemMessage(
            content=f"현재 시각은 {now}입니다. "
            "너는 도움이 되는 경제 전문가 입니다. 주어진 정보를 근거로 사용자의 질문에 적합한 경제 분석 보고서를 작성하세요. "
            "사용자의 현재 포트폴리오, 관심사, 이전 요청기록을 참고하여 보고서를 작성하세요."
            "사용자의 보고서 양식에 맞춰 보고서를 작성하세요. "
            f"보고서 양식:\n{user_profile['user_preference']}  "
            "항상 주어진 정보를 근거로 삼고 가능한 한 정확하고 자세하게 서술하세요. "
            "사용자의 질문에 따라 결정한 세부 조사항목과 세부 조사항목 정보를 사용하세요. "
            "구체적인 수치, 기술, 상황 정보가 보고서에 포함되어야 합니다. 수치 정보는 가능하면 퍼센트보다는 구체적인 수치를 포함하세요. "
            "문체는 ~입니다 체가 아닌 ~다 를 사용하세요. "
            # "사용자의 질문에 따른 보고서를 작성할 때 논리적으로 근거가 부족하거나 반례를 조사해야 할 경우 '→ **추가로 검색해야 할 명확한 검색 키워드**' 한 만 보고서 끝에 출력하세요. "
        ),
        HumanMessage(content=user_prompt + user_profile_prompt),
    ]

    response = llm.invoke(messages).content.strip()
    return response


def self_ask_agent(
    user_query: str,
    user_profile,
    reranker_model,
    embedding_model,
    open_key,
    top_k=20,
    top_n=5,
    max_count=10,
    self_ask_iter=3,
):
    llm = ChatOpenAI(model="gpt-4o", temperature=1, api_key=open_key)
    current_question = user_query
    final_answer = None
    print(f"\n Step 0: 질문 - {current_question}")
    expand_query_result = expand_query2(llm, current_question, user_profile)

    contexts, valid_subqueries = [], []
    external_db, all_response = [], []
    for subquery in expand_query_result:
        print(f"\n Subquery: 질문 - {subquery}")
        documents = search_googlenews(subquery, max_count=max_count)
        if documents is None:
            continue
        print("In process of retrieval")
        chunks = chunk_documents(documents)
        external_db.extend(chunks)

        results = retrieval_web(
            subquery, chunks, reranker_model, embedding_model, top_k=top_k, top_n=top_n
        )
        valid_subqueries.append(subquery)
        contexts.append(results)

        print(f"Writing report for {subquery}")
        all_response.append(
            llm_call_single(llm, current_question, subquery, user_profile, results)
        )

    # final_answer = llm_call(llm, user_query, valid_subqueries, contexts)

    # for step in range(self_ask_iter):  # max 5 iteration
    #   flag = decide_or_generate_subquestion(llm, user_query, memory.get_combined_context())
    #   if not flag[0]:
    #     current_question = flag[1]
    #     print(f"\n Step {step+1}: 질문 - {current_question}")
    #     context, source = search_web(current_question)
    #     for c, s in zip(context, source):
    #       memory.add_context(c, s)
    #   else:
    #     break

    # print("충분한 정보 확보. 최종 응답 생성 중...")
    # final_answer = llm_call(llm, user_query, memory.get_combined_context())
    # print(final_answer)

    final_answer = "\n\n".join(all_response)
    print(final_answer)

    return final_answer, contexts, external_db


# open_key = userdata.get("OPENAPI_KEY") # your api key

if __name__ == "__main__":
    api_key = getpass.getpass("OpenAI API Key 입력: ")
    user_query = getpass.getpass("user_query 입력: ")
    embedding_model = get_embedding_model()
    reranker_model, reranker_tokenizer = get_reranker_model()
    user_profile = get_default_profile()
    print("모델 로딩 완료")

    response, contexts, external_db = self_ask_agent(
        user_query, user_profile, reranker_model, embedding_model, api_key
    )
    print(response)
