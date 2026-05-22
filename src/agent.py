"""Orchestrates fetch -> PDF/abstract text -> chunk -> FAISS -> Structured Output."""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any, Dict, List, Set, Tuple

from .database import ChunkStore, get_active_store, set_active_store
from .fetch import fetch_papers
from .llm import (
    format_segments, generate_query_plan, detect_contradictions, 
    evaluate_report, refine_report
)
from .pdf_reader import chunk_text, load_paper_document

logger = logging.getLogger(__name__)

DEFAULT_PAPER_LIMIT = 10
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K = 30  # Candidate pool for sampling
FOLLOW_UP_K = 15
RELEVANCE_THRESHOLD = 0.3  # Adjusted to be more inclusive during indexing
BORDERLINE_THRESHOLD = 0.2
MAX_CHUNKS_PER_PAPER = 3  # Slightly increased for better depth

def _extract_topic_keywords(topic: str) -> Set[str]:
    """Extracts unique lowercase keywords, splitting on hyphens and spaces."""
    stop_words = {"in", "on", "the", "and", "for", "with", "of", "to", "from", "at", "by", "an", "is", "it"}
    # Split on non-alphanumeric to catch "multi-agent" as "multi", "agent"
    words = re.split(r'[^a-zA-Z0-9]+', topic.lower())
    return {w for w in words if w not in stop_words and len(w) > 2}

def _calculate_relevance(text: str, keywords: Set[str]) -> float:
    """Calculates a simple relevance score based on keyword presence and negative filters."""
    if not keywords: return 1.0 # No keywords to check against
    text_lower = text.lower()
    
    # 1. Positive matches
    matches = sum(1 for kw in keywords if kw in text_lower)
    score = matches / len(keywords)
    
    # 2. Negative filtering (Anti-topic detection)
    # Using word boundaries to avoid false positives (e.g., 'grade' in 'gradient')
    bad_patterns = [r'\bstudent\b', r'\bclassroom\b', r'\btutor\b', r'\bgrade\b']
    if not any(kw in ["education", "tutor", "learn", "student"] for kw in keywords):
        if sum(1 for pat in bad_patterns if re.search(pat, text_lower)) >= 2:
            return 0.0 # Rejection of unrelated domain
            
    return score

def _group_chunks_by_theme(chunks: List[Dict[str, Any]]) -> str:
    """Groups chunks into logical themes for better LLM reasoning."""
    themes = {
        "METHODOLOGIES": [],
        "APPLICATIONS": [],
        "LIMITATIONS/GAPS": [],
        "KEY FINDINGS": []
    }
    
    for c in chunks:
        text = c.get("text", "").lower()
        sec = c.get("section", "").lower()
        
        if any(k in sec or k in text for k in ["method", "approach", "architecture", "design", "algorithm"]):
            themes["METHODOLOGIES"].append(c)
        elif any(k in sec or k in text for k in ["apply", "use case", "real-world", "implementation", "deployment"]):
            themes["APPLICATIONS"].append(c)
        elif any(k in sec or k in text for k in ["limit", "gap", "future", "challenge", "problem"]):
            themes["LIMITATIONS/GAPS"].append(c)
        elif any(k in sec or k in text for k in ["result", "finding", "eval", "conclusion", "show"]):
            themes["KEY FINDINGS"].append(c)
            
    context_str = ""
    for theme, theme_chunks in themes.items():
        if theme_chunks:
            context_str += f"\n[{theme}]\n"
            # Limit to top 4 per theme to keep context concise but rich
            for tc in theme_chunks[:4]:
                context_str += f"- {tc.get('text')}\n"
                
    return context_str.strip()


def _configure_logging() -> None:
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

def _process_paper_concurrently(p: dict) -> dict:
    title = p.get("title", "")
    link = p.get("link", "")
    pdf_link = p.get("pdf_link", "")
    summary = p.get("summary", "")
    source = (p.get("source") or "unknown").strip().lower()
    topic_keywords = p.get("_topic_keywords", set())

    # Quick summary check
    # We trust the initial search engine results more now, so we skip the summary relevance check
    # which was causing false positives on short or highly specific queries like "4.0".

    doc = load_paper_document(pdf_link or link, summary)
    full_text = doc.get("full_text") or ""
    if not full_text.strip():
        logger.warning("Paper text extraction failed or returned empty: %s", title[:40])
        return {"success": False, "source": source}

    sections = doc.get("sections") or {}
    section_items = list(sections.items())
    has_structured_text = any((sec_body or "").strip() for _, sec_body in section_items)

    # Zero-Noise PDF artifact check
    alnum_ratio = sum(1 for c in full_text if c.isalnum() or c.isspace()) / len(full_text)
    if alnum_ratio < 0.8:
        logger.warning("Paper rejected due to high noise/artifacts in full text: %s", title[:40])
        return {"success": False, "source": source}

    batches = []

    if has_structured_text:
        for sec_name, sec_body in section_items:
            sec_body = (sec_body or "").strip()
            if not sec_body:
                continue
            sec_chunks = chunk_text(sec_body, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            # Filter chunks: remove noise
            clean_chunks = [ch for ch in sec_chunks if len(ch) > 100 and sum(1 for char in ch if char.isalnum()) / len(ch) > 0.8]
            if clean_chunks:
                batches.append({"sec_name": sec_name, "chunks": clean_chunks})
    else:
        chunks = chunk_text(full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        clean_chunks = [ch for ch in chunks if len(ch) > 100 and sum(1 for char in ch if char.isalnum()) / len(ch) > 0.8]
        if clean_chunks:
            batches.append({"sec_name": "document", "chunks": clean_chunks})
            
    return {
        "success": True,
        "title": title,
        "link": link,
        "source": source,
        "batches": batches
    }



def retrieve_diverse_chunks(query: str, k: int, max_per_paper: int = 5) -> List[Dict[str, Any]]:
    """Retrieves chunks for a query while limiting segments from the same paper for diversity."""
    store = get_active_store()
    if store is None or store.size == 0:
        logger.warning("Retrieval attempted but store is empty or inactive.")
        return []
    
    logger.info("Retrieving diverse chunks for query: %r (target_k=%d)", query, k)
    
    # Retrieve more candidates than requested to allow for diversification
    candidates = store.search(query, k=max(k * 3, 50))
    if not candidates:
        return []

    # Group by paper title and limit chunks per paper
    diverse_results: List[Dict[str, Any]] = []
    paper_counts: Dict[str, int] = {}
    
    for chunk in candidates:
        title = chunk.get("paper_title", "Unknown")
        count = paper_counts.get(title, 0)
        
        if count < max_per_paper:
            diverse_results.append(chunk)
            paper_counts[title] = count + 1
            
        if len(diverse_results) >= k:
            break
            
    logger.info("Diverse retrieval complete: %d segments from %d papers.", len(diverse_results), len(paper_counts))
    return diverse_results


def retrieve_for_query(query: str, k: int = FOLLOW_UP_K) -> List[Dict[str, Any]]:
    """Legacy wrapper for retrieve_diverse_chunks."""
    return retrieve_diverse_chunks(query, k)


def run_agent(
    topic: str, 
    limit: int = DEFAULT_PAPER_LIMIT, 
    mode: str = "student", 
    angle: str = "methods",
    gemini_api_key: str = None,
    semantic_scholar_key: str = None
) -> str:
    """Agentic Research Pipeline: Plan -> Fetch -> Index -> Retrieve -> Validate -> Reflect."""
    _configure_logging()
    topic = (topic or "").strip()
    if not topic:
        return "Topic is empty."

    # 1. QUERY PLANNING
    logger.info("Step 1: Generating Research Plan for: %r", topic)
    subtopics = generate_query_plan(topic, gemini_api_key)
    logger.info("Plan generated with %d subtopics: %s", len(subtopics), subtopics)

    # 2. MULTI-SOURCE RETRIEVAL & INDEXING
    logger.info("Step 2: Fetching and Indexing Papers...")
    store = ChunkStore()
    set_active_store(store)
    
    all_papers = []
    seen_titles = set()
    
    # Concurrent fetching for each subtopic
    with ThreadPoolExecutor(max_workers=5) as fetch_executor:
        futures = {fetch_executor.submit(fetch_papers, q, limit=max(3, limit//len(subtopics)), semantic_scholar_key=semantic_scholar_key): q for q in subtopics}
        for future in as_completed(futures):
            papers = future.result()
            for p in papers:
                if p["title"] not in seen_titles:
                    all_papers.append(p)
                    seen_titles.add(p["title"])

    if not all_papers:
        return "No research papers found for the planned subtopics."

    # Process and Index
    keywords = _extract_topic_keywords(topic)
    seen_signatures = set()
    with ThreadPoolExecutor(max_workers=10) as process_executor:
        future_to_paper = {process_executor.submit(_process_paper_concurrently, p): p for p in all_papers}
        for future in as_completed(future_to_paper):
            res = future.result()
            if not res.get("success"): continue
            
            title, link, source, batches = res["title"], res["link"], res["source"], res["batches"]
            current_paper_chunks = 0
            
            for batch in batches:
                sec_name = batch["sec_name"]
                unique_chunks = []
                for ch in batch["chunks"]:
                    if current_paper_chunks >= MAX_CHUNKS_PER_PAPER: break # DIVERSITY CONTROL
                    
                    rel_score = _calculate_relevance(ch, keywords)
                    if rel_score < RELEVANCE_THRESHOLD: continue # STRICT VALIDATION
                    
                    sig = (title, sec_name, ch[:200])
                    if sig in seen_signatures: continue
                    seen_signatures.add(sig)
                    unique_chunks.append(ch)
                    current_paper_chunks += 1
                
                if unique_chunks:
                    store.add_chunks(unique_chunks, title, link, sec_name, source)

    if store.size == 0:
        return "Failed to index any high-quality research segments."

    # 3. TARGETED RETRIEVAL & TENSION DETECTION
    logger.info("Step 3: Diverse Retrieval & Tension Detection...")
    retrieved_chunks = retrieve_diverse_chunks(topic, k=20, max_per_paper=MAX_CHUNKS_PER_PAPER)
    
    # Borderline inclusion (limit to 2)
    borderline_count = 0
    if len(retrieved_chunks) < 10:
        candidates = store.search(topic, k=50)
        for c in candidates:
            if borderline_count >= 2: break
            score = _calculate_relevance(c["text"], keywords)
            if BORDERLINE_THRESHOLD <= score < RELEVANCE_THRESHOLD:
                retrieved_chunks.append(c)
                borderline_count += 1

    tensions = detect_contradictions(retrieved_chunks, gemini_api_key)

    # 4. REPORT GENERATION
    logger.info("Step 4: Generating initial report...")
    themed_context = _group_chunks_by_theme(retrieved_chunks)
    report = format_segments(retrieved_chunks, topic=topic, mode=mode, angle=angle, gemini_api_key=gemini_api_key, themed_context=themed_context, tensions=tensions)

    # 5. SELF-REFLECTION & REFINEMENT
    logger.info("Step 5: Self-Reflection & Refinement Pass...")
    feedback = evaluate_report(report, gemini_api_key)
    logger.info("Self-Reflection Quality Score: %s/10", feedback.get("quality_score"))
    
    if feedback.get("quality_score", 10) < 9:
        refined_report = refine_report(report, feedback, gemini_api_key)
        return refined_report

    return report


def answer_follow_up(query: str, gemini_api_key: str = None) -> str:
    """Retrieves chunks specifically for a follow-up query and returns them formatted."""
    logger.info("Follow-up question received: %r", query)
    chunks = retrieve_diverse_chunks(query, k=FOLLOW_UP_K, max_per_paper=MAX_CHUNKS_PER_PAPER)
    logger.info("Retrieved %d segments for follow-up.", len(chunks))
    return format_segments(chunks, topic=query, gemini_api_key=gemini_api_key)
