"""
DriveLegal – rag/chain.py
RAG chain using local Ollama (Mistral) with offline fallback and fuzzy keyword matching.
"""

import json
import logging
import os
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from rag.challan import lookup_violation_records
from rag.retriever import retrieve

logger = logging.getLogger("drivelegal.chain")
CACHE_PATH = os.path.join("data", "cache.json")

SYSTEM_TEMPLATE = """You are DriveLegal, an AI road safety and traffic law assistant.
{gps_context_line}Use ONLY the context provided below to answer the question.
If the context does not contain enough information, say:
  "I could not find specific data for your location. Please verify with the official traffic authority."
Never invent fine amounts or law sections.
When the context includes international treaty rules (Vienna Convention on Road
Traffic 1968, UN Model Road Safety Legislation, EU Directive 2015/413), cite them
FIRST as the baseline applicable to all signatory nations, then add any country-
specific or regional rules that differ or supplement the baseline.
Always clarify the tier of each rule you cite:
  • TREATY — international baseline (Vienna Convention, UN Model, EU Directive)
  • NATIONAL — country-level law
  • STATE/CITY — sub-national or local rule
If a fine is mentioned, state it with the disclaimer that amounts are indicative
and subject to local enforcement.

Context:
{context}

Question: {question}

Answer clearly and concisely in 2-5 sentences."""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "gps_context_line"],
    template=SYSTEM_TEMPLATE,
)


def _load_cache() -> list:
    if not os.path.exists(CACHE_PATH):
        return []
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    """Normalize text for better matching."""
    replacements = {
        "drinking and driving": "drunk driving",
        "drinking & driving": "drunk driving",
        "drink driving": "drunk driving",
        "drunken driving": "drunk driving",
        "two wheeler": "bike",
        "two-wheeler": "bike",
        "motorbike": "bike",
        "motorcycle": "bike",
        "four wheeler": "car",
        "four-wheeler": "car",
        "seatbelt": "seat belt",
        "licence": "license",
        "no helmet": "helmet",
        "without helmet": "helmet",
        "jumping red light": "red light",
        "running red light": "red light",
        "jump signal": "red light",
        "traffic signal": "signal",
        "traffic light": "signal",
        "mobile phone": "mobile",
        "cell phone": "mobile",
        "using phone": "mobile",
        "puc certificate": "pollution",
        "pollution certificate": "pollution",
        "rc book": "registration",
        "vehicle registration": "registration",
        "driving licence": "license",
        "driving license": "license",
    }
    text = text.lower()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _offline_fallback(
    query: str,
    city: str = "",
    state: str = "",
    country: str = "India",
) -> Dict[str, Any]:
    """Fallback logic: Search Qdrant directly. Use cache only if Qdrant fails."""
    normalized_query = _normalize(query)

    # First resort: search Qdrant directly in offline mode
    try:
        docs, top_score = retrieve(
            normalized_query,
            k=3,
            city=city,
            state=state,
            country=country,
        )
        if docs:
            return {
                "answer": (
                    f"{docs[0].page_content[:300]}... "
                    f"\n\n(Note: AI is offline. This is the closest raw law section found.)"
                ),
                "sources": docs,
                "offline_fallback": True,
                "top_retrieval_score": top_score,
            }
    except Exception as e:
        logger.error(f"Qdrant fallback failed: {e}")

    # Second resort: keyword match against cache
    cache = _load_cache()
    best, best_score = None, 0
    for entry in cache:
        keywords = entry.get("keywords", [])
        score = sum(1 for kw in keywords if _normalize(kw) in normalized_query)
        if score > best_score:
            best, best_score = entry, score

    if best and best_score > 0:
        return {
            "answer": best["answer"],
            "sources": [],
            "offline_fallback": True,
            "top_retrieval_score": 0.0,
        }

    return {
        "answer": (
            "I could not find specific information for your query. "
            "Please verify with the official RTO or visit https://morth.nic.in"
        ),
        "sources": [],
        "offline_fallback": True,
        "top_retrieval_score": 0.0,
    }


def get_answer(
    query: str,
    city: str,
    state: str,
    country: str = "India",
    gps_context: str = "",
) -> Dict[str, Any]:
    """Run RAG pipeline. Falls back to offline cache if Ollama (Mistral) unavailable.

    Args:
        query:       The user's natural-language question.
        city:        City resolved from GPS or manual dropdown.
        state:       State resolved from GPS or manual dropdown.
        country:     Country (default India).
        gps_context: Optional human-readable location context injected into the
                     system prompt (e.g. "Pune, Maharashtra, India"). When provided,
                     Mistral is explicitly told the user's physical location.
    """
    # Build the GPS context line — blank if no GPS fix available
    gps_context_line = (
        f"The user is currently located near {gps_context}. "
        "Tailor your answer to local enforcement where possible.\n"
        if gps_context
        else ""
    )
    try:
        docs, top_score = retrieve(query, k=5, city=city, state=state, country=country)

        if not docs:
            return {
                "answer": "I could not find specific data for your location. Please verify with the official traffic authority.",
                "sources": [],
                "offline_fallback": False,
                "top_retrieval_score": top_score,
            }

        context = "\n\n".join(
            "[{tier} | {region} | {section}]\n{text}".format(
                tier=(d.metadata.get("corpus_tier") or "regional").upper(),
                region=d.metadata.get("region", "?"),
                section=d.metadata.get("law_section", ""),
                text=d.page_content,
            )
            for d in docs
        )

        records = lookup_violation_records(query, state=state)
        if records:
            record_context = "\n".join(
                " | ".join(f"{key}: {value}" for key, value in record.items() if value)
                for record in records
            )
            context += (
                "\n\n[STRUCTURED DATASET RECORDS | use exact values]\n"
                + record_context
            )

        llm = ChatOllama(model="mistral", temperature=0)
        prompt = QA_PROMPT.format(
            context=context,
            question=query,
            gps_context_line=gps_context_line,
        )
        response = llm.invoke(prompt)

        return {
            "answer": response.content.strip(),
            "sources": docs,
            "offline_fallback": False,
            "top_retrieval_score": top_score,
        }

    except Exception as exc:
        logger.warning(f"LLM unavailable, using offline fallback. Reason: {exc}")
        return _offline_fallback(query, city=city, state=state, country=country)