import tiktoken
import math
import re
from datetime import datetime
from datetime import datetime, timedelta
from langchain.schema import Document
from email.utils import parsedate_to_datetime
import time

# 외부 db 사용시 timestamp 연산으로 sql query 사용
def filter_documents_by_ttl(documents, ttl_days:int = 180):
  valid_docs = []
  ttl_seconds = ttl_days * 24 * 60 * 60
  now_ts = int(time.time())

  for doc in documents:
      timestamp = doc.metadata.get("published_ts", None)
      if timestamp is None:
        continue
      if now_ts - timestamp <= ttl_seconds:
        valid_docs.append(doc)
  return valid_docs

encoding = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(encoding.encode(text))

def decay_for_half_life(half_life, unit="days"):
  if unit == "days":
    return math.log(2) / half_life
  elif unit == "seconds":
    return math.log(2) / half_life

def recency_weight(dt_doc, now=None, decay_rate=0.05):
  if now is None:
    if dt_doc.tzinfo is not None:
      now = datetime.now(dt_doc.tzinfo)
    else:
      now = datetime.now()
  delta = (now - dt_doc).days
  return math.exp(-decay_rate * delta)

def strip_prefix(documents):
    pattern = r"^\[\d{4}-\d{2}-\d{2}\]passage:\s*"
    for doc in documents:
      doc.page_content = re.sub(pattern, "", doc.page_content)
    return documents

def strip_prefix_time(documents):
    pattern = r"^\[\d{4}-\d{2}-\d{2}\]\s*"
    for doc in documents:
      doc.page_content = re.sub(pattern, "", doc.page_content)
    return documents

#threshold 및 half_life 조정 필요
def judge_websearch(docs_with_scores, threshold=0.5, half_life = 90):
  adjusted_results = []
  for doc, score in docs_with_scores:
    dt = doc.metadata.get("published")
    decay_rate = decay_for_half_life(half_life)
    weight = recency_weight(dt, decay_rate=decay_rate)

    adjusted_score = score * weight

    adjusted_results.append((doc, adjusted_score))
    adjusted_results.sort(key=lambda x: x[1], reverse=True)
    scores = [score for _, score in adjusted_results[:10]]
    if np.mean(scores) > threshold:
      return True, np.mean(scores)
    else:
      return False, np.mean(scores)
  return adjusted_results