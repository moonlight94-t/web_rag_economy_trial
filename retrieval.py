from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import DistanceStrategy
from util import strip_prefix_time


def chunk_documents(documents, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def retrieval_web(
    user_query,
    chunks,
    reranker_model,
    embedding_model,
    top_k=20,
    top_n=5,
):
    # chunks = chunk_documents(documents)
    for doc in chunks:
        doc.page_content = "passage: " + doc.page_content
    faiss_index = FAISS.from_documents(
        chunks, embedding_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    mmr_retriever = faiss_index.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 50,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },  # k=2개 반환, fetch_k=4개 후보 중 MMR 적용, 관련성에 더 비중(0.7)
    )
    faiss_retrieve = mmr_retriever.invoke("query: " + user_query)
    # faiss_retrieve = faiss_index.similarity_search('query: '+user_query, k=top_k)

    for doc in faiss_retrieve:
        doc.page_content = doc.page_content.removeprefix("passage:").strip()
        doc.page_content = (
            f"[{doc.metadata.get('published').strftime('%Y-%m-%d')}]" + doc.page_content
        )

    reranker = CrossEncoderReranker(model=reranker_model, top_n=top_n)
    reranked = reranker.compress_documents(query=user_query, documents=faiss_retrieve)

    reranked = strip_prefix_time(reranked)
    return reranked
