"""Fetch paper metadata from arXiv + Semantic Scholar."""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_HEADERS = {"User-Agent": "research-ai/1.0 (+https://localhost)"}


def _normalize_query(topic: str) -> str:
    t = topic.strip()
    if not t:
        return ""
    return re.sub(r"\s+", " ", t)


def _fetch_arxiv(topic: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Query arXiv and return normalized paper records."""
    topic = _normalize_query(topic)
    if not topic:
        return []

    logger.info("Querying arXiv for topic: %r (limit=%d)", topic, limit)
    # all: field searches title, abstract, comments
    q = quote(f"all:{topic}", safe="")
    url = f"{ARXIV_API}?search_query={q}&max_results={limit}&sortBy=relevance&sortOrder=descending"

    try:
        response = requests.get(url, timeout=30, headers=DEFAULT_HEADERS)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("arXiv API request failed: %s", e)
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error("Failed to parse arXiv XML response: %s", e)
        return []
        
    papers: List[Dict[str, Any]] = []

    for entry in root.findall(f"{NS}entry"):
        title_el = entry.find(f"{NS}title")
        summary_el = entry.find(f"{NS}summary")
        id_el = entry.find(f"{NS}id")
        if title_el is None or id_el is None:
            continue
        title = (title_el.text or "").strip()
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        link = (id_el.text or "").strip()
        if not title or not link:
            continue
        papers.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "pdf_link": "",
                "source": "arxiv",
            }
        )

    logger.info("arXiv returned %d papers.", len(papers))
    return papers


def _fetch_semantic_scholar(topic: str, limit: int = 10, api_key: str = None) -> List[Dict[str, Any]]:
    """Query Semantic Scholar and return normalized paper records."""
    topic = _normalize_query(topic)
    if not topic:
        return []

    logger.info("Querying Semantic Scholar for topic: %r (limit=%d)", topic, limit)
    params = {
        "query": topic,
        "limit": max(1, min(limit, 100)),
        "fields": "title,abstract,url,openAccessPdf",
    }
    headers = dict(DEFAULT_HEADERS)
    api_key = api_key or (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
    if api_key:
        logger.info("Using Semantic Scholar API Key.")
        headers["x-api-key"] = api_key

    # A short retry helps when the API briefly rate-limits anonymous requests.
    for attempt in range(2):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_API,
                params=params,
                timeout=30,
                headers=headers,
            )
            if response.status_code == 429 and attempt == 0:
                logger.warning("Semantic Scholar rate-limited (429). Retrying in 1s...")
                time.sleep(1.0)
                continue
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 0:
                logger.warning("Semantic Scholar attempt 1 failed: %s. Retrying...", e)
                continue
            logger.error("Semantic Scholar API request failed after 2 attempts: %s", e)
            return []

    try:
        payload = response.json() if response.content else {}
    except Exception as e:
        logger.error("Failed to parse Semantic Scholar JSON: %s", e)
        return []
        
    rows = payload.get("data") or []
    papers: List[Dict[str, Any]] = []
    for row in rows:
        title = (row.get("title") or "").strip()
        summary = (row.get("abstract") or "").strip()
        link = (row.get("url") or "").strip()
        open_access = row.get("openAccessPdf") or {}
        pdf_link = (open_access.get("url") or "").strip()
        if not title:
            continue
        papers.append(
            {
                "title": title,
                "summary": summary,
                "link": link or pdf_link,
                "pdf_link": pdf_link,
                "source": "semantic_scholar",
            }
        )
    logger.info("Semantic Scholar returned %d papers.", len(papers))
    return papers


def fetch_papers(topic: str, limit: int = 10, semantic_scholar_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch papers for topic. Prioritizes Semantic Scholar, and fills remaining quota from arXiv.
    """
    topic = _normalize_query(topic)
    if not topic:
        logger.warning("Empty topic provided to fetch_papers.")
        return []

    papers = _fetch_semantic_scholar(topic, limit=limit, api_key=semantic_scholar_key)
    
    remaining_limit = limit - len(papers)
    if remaining_limit > 0:
        if len(papers) == 0:
            logger.info("Semantic Scholar returned no results; falling back entirely to arXiv.")
        else:
            logger.info(f"Fetching remaining {remaining_limit} papers from arXiv to reach limit.")
            
        arxiv_papers = _fetch_arxiv(topic, limit=remaining_limit)
        papers.extend(arxiv_papers)

    return papers
