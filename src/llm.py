import re
import random
from typing import Any, Dict, List, Set
from collections import Counter, defaultdict

try:
    import google.generativeai as genai
except ImportError:
    genai = None

def calculate_confidence(chunks: List[Dict[str, Any]]) -> str:
    """Calculates confidence level based on source diversity and volume."""
    papers = set(c.get("paper_title", "Unknown") for c in chunks if c.get("paper_title") != "Unknown")
    num_papers = len(papers)
    
    if num_papers >= 8: return "High"
    if num_papers >= 4: return "Medium"
    return "Low"

def clean_text(text: str) -> str:
    """Aggressively cleans PDF artifacts, noise, and broken formatting."""
    if not text:
        return ""
    
    # 1. Fix broken hyphenation (e.g., "fea- tures" -> "features")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # 2. Fix common PDF character glitches
    text = text.replace('f i', 'fi').replace('f l', 'fl').replace('f f', 'ff').replace('f t', 'ft')
    
    # 3. Remove Footer/Header noise (e.g., "Page 1 of 10", "Download from...", "DOI:...")
    text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'(?i)proceedings of the.*?\d{4}', '', text)
    text = re.sub(r'doi:\s+\S+', '', text)
    
    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Remove citation noise (e.g. [1, 2], (Smith et al., 2020))
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\(\w+ et al\., \d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    
    return text.strip()

def split_sentences(text: str) -> List[str]:
    """Nuclear Quality Gate: Only allows perfect, complete, human-readable sentences."""
    text = clean_text(text)
    # Temporary placeholder for periods in abbreviations
    text = re.sub(r'\bet al\.', 'et_al', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(e\.g|i\.e|fig|vs)\.', r'\1_dot', text, flags=re.IGNORECASE)
    
    # Split on terminal punctuation followed by capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    cleaned = []
    for s in sentences:
        s = s.replace('et_al', 'et al.').replace('_dot', '.').strip()
        
        # 1. Length & Structure Gate
        if len(s) < 100: continue # Strictly reject fragments
        if len(s) > 500: continue # Reject run-on junk
        if not s[0].isupper() or not s.endswith(('.', '!', '?')): continue
        
        # 2. OCR Quality Gate (Nuclear)
        # Reject if isolated characters exist (broken words)
        if re.search(r'\b[b-hj-km-np-tv-z]\b', s): continue 
        # Reject if rithms, onebox, or other corrupted fragments are found
        if any(bad in s.lower() for bad in ["rithms", "onebox", "procee", "vol.", "pp.", "et al"]): continue
        
        # 3. Cleanliness Gate
        alnum_ratio = sum(1 for c in s if c.isalnum() or c.isspace()) / len(s)
        if alnum_ratio < 0.9: continue # Strictly reject symbol-heavy or messy OCR
            
        cleaned.append(s)
    return cleaned

def generate_heuristic_insight(chunk: Dict[str, Any], key_idea: str) -> str:
    """
    Synthesizes technical interpretation: what it means, why it matters, and where it can be used.
    """
    text = chunk.get("text", "").lower()
    section = chunk.get("section", "General").lower()
    
    # Interpretation logic
    if any(w in section for w in ["method", "approach", "architecture"]):
        meaning = "This segment outlines a foundational architectural design or algorithmic logic used to solve the core problem."
        utility = "It is essential for engineers looking to replicate or build upon the system's structural framework."
    elif any(w in section for w in ["result", "evalu", "experiment"]):
        meaning = "This data provides empirical evidence and benchmarks, quantifying the model's success against standard baselines."
        utility = "It can be used to justify resource allocation or to identify specific performance bottlenecks in real-time apps."
    elif "limit" in section or "challenge" in section:
        meaning = "This identifies critical operational constraints or edge cases where the current approach may fail."
        utility = "Understanding this is vital for risk assessment in safety-critical domains like healthcare or autonomous systems."
    else:
        meaning = "This provides necessary theoretical context and definitions required to navigate the research domain effectively."
        utility = "It serves as a conceptual roadmap for newcomers trying to align with the paper's specific terminology."

    importance = "It bridges the gap between theoretical claims and practical implementation by highlighting real-world trade-offs."
    if "sota" in text or "outperform" in text:
        importance = "This is high-impact as it demonstrates a significant leap over existing state-of-the-art benchmarks."

    return f"{meaning} {importance} {utility}"

def extract_meaningful_concepts(chunks: List[Dict[str, Any]], top_n: int = 5) -> List[str]:
    """
    Heuristic to find recurring technical concepts across papers.
    """
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "their", "will", "been",
        "which", "were", "where", "model", "system", "study", "research", "paper",
        "approach", "results", "analysis", "using", "between", "these", "those",
        "about", "into", "highly", "provide", "based", "proposed", "performance",
        "significant", "existing", "development", "application", "different"
    }
    
    paper_texts = defaultdict(str)
    for c in chunks:
        paper_texts[c.get("paper_title", "Unknown")] += " " + c.get("text", "").lower()
        
    concepts = Counter()
    for text in paper_texts.values():
        clean = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        words = clean.split()
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        
        valid_bigrams = set()
        for b in bigrams:
            parts = b.split()
            if len(parts) == 2 and all(p not in stop_words and len(p) > 3 for p in parts):
                valid_bigrams.add(b)
        
        for b in valid_bigrams:
            concepts[b] += 1
            
    return [c for c, count in concepts.most_common(top_n) if count >= 2]

def analyze_user_paper(sections: Dict[str, str]) -> str:
    """
    Detailed, beginner-friendly critique of a research paper.
    Uses expanded templates for Summary, Strengths, Gaps, and Scoring.
    """
    intro = sections.get("intro", "")
    lit_review_text = sections.get("lit_review", "")
    method = sections.get("method", "")
    results = sections.get("results", "")
    discussion_text = sections.get("discussion", "")
    conclusion = sections.get("conclusion", "")
    full_text = f"{intro} {lit_review_text} {method} {results} {discussion_text} {conclusion}".lower()
    
    # 1. Validation
    total_len = len(full_text.strip())
    if total_len < 1000:
        return "## Insufficient Content\n- The extracted text is too short for a meaningful heuristic analysis. Ensure the PDF has selectable text."

    has_core = len(intro) > 100 or len(method) > 200 or len(conclusion) > 100
    if not has_core:
        return "## Analysis Blocked\n- No major academic sections detected. Use standard headers like 'Introduction' or 'Methods'."

    # 2. Detailed Summary (4-6 lines)
    summary_sentences = split_sentences(intro)[:3] if intro else ["The paper investigates technical concepts within its domain."]
    problem_hint = "The core problem addressed involves optimizing system performance or addressing specific technical constraints identified in the introduction."
    approach_hint = f"The proposed approach utilizes {'methodology-specific logic' if not method else 'the described methodology'} to achieve its objectives."
    importance_hint = "This work is important as it contributes to the broader understanding of efficiency and robustness in this research area."
    summary_block = f"{' '.join(summary_sentences)}\n\n**Problem & Approach**: {problem_hint} {approach_hint}\n\n**Importance**: {importance_hint}"

    # 3. Strengths & Weaknesses (Detailed)
    strengths = []
    if len(method) > 1500:
        strengths.append("**Technical Depth**: The methodology is comprehensive and well-documented. *Utility*: This allows other researchers to replicate the findings and verify the technical validity of the results.")
    if any(kw in full_text for kw in ["baseline", "compare", "performance"]):
        strengths.append("**Comparative Analysis**: The paper includes benchmarking against existing standards. *Utility*: This provides a clear quantitative measure of how much better the proposed solution is compared to current state-of-the-art.")
    if len(intro) > 1000:
        strengths.append("**Contextual Foundation**: The introduction provides a strong background. *Utility*: This helps beginners understand the 'why' behind the research before diving into the complex technical details.")
    if not strengths: strengths.append("**Structured Format**: The paper follows a logical academic structure, which makes it easier to navigate for readers.")

    weaknesses = []
    if len(method) < 1000:
        weaknesses.append("**Methodological Brevity**: The technical implementation details are somewhat sparse. *Impact*: Readers may find it difficult to fully grasp the 'how' or replicate the experiments accurately.")
    if "limit" not in full_text:
        weaknesses.append("**Omitted Limitations**: The paper does not explicitly discuss constraints. *Impact*: Without knowing the edge cases, practitioners might over-rely on the system in scenarios where it is likely to fail.")
    if len(results) < 500:
        weaknesses.append("**Insufficient Empirical Evidence**: The results section lacks granular data. *Impact*: The claims made in the conclusion are not fully supported by a broad range of test cases or statistical significance.")

    # 4. Actionable Suggestions (Problem/Fix/Impact)
    suggestions = []
    if len(method) < 1500:
        suggestions.append("**Problem**: Methodological implementation details are sparse.\n- **Fix**: Expand the Methodology with algorithmic pseudocode or formal mathematical proofs.\n- **Impact**: Enhances technical rigor and allows for precise replication by other researchers.")
    
    if any(kw in full_text for kw in ["learning", "training", "dataset"]):
        if "baseline" not in full_text:
            suggestions.append("**Problem**: Lack of standardized ML benchmarking.\n- **Fix**: Compare accuracy and F1-score against SOTA baselines on a public dataset like ImageNet or GLUE.\n- **Impact**: Validates that the proposed model offers a statistically significant performance gain.")
    
    if any(kw in full_text for kw in ["latency", "throughput", "overhead", "distributed"]):
        if "scalability" not in full_text:
            suggestions.append("**Problem**: Architectural scalability constraints are not addressed.\n- **Fix**: Conduct latency tests under varying concurrent loads to identify the system's saturation point.\n- **Impact**: Ensures the system is production-ready for high-throughput enterprise environments.")

    if "limit" not in full_text:
        suggestions.append("**Problem**: Operational constraints and edge cases are omitted.\n- **Fix**: Add a dedicated 'Limitations' subsection discussing potential failure modes or data biases.\n- **Impact**: Provides a balanced view and helps users avoid misapplying the research in risky scenarios.")

    # 5. Research Gaps (Specific Template)
    gaps = []
    if "scalability" not in full_text:
        gaps.append("**Gap: Scalability in High-Throughput Environments**\n\n"
                    "**Explanation**: The current analysis primarily evaluates the system's performance on isolated or small-scale datasets. It does not account for how the architecture behaves when data volume increases by orders of magnitude.\n\n"
                    "**Why it matters**: In real-world applications like financial trading or social media monitoring, a lack of scalability can lead to system crashes or unacceptable latency spikes.\n\n"
                    "**Future Work**: Conduct a stress test using distributed computing frameworks to identify the saturation point of the current model.")
    
    if "robustness" not in full_text:
        gaps.append("**Gap: Resilience Against Adversarial and Noisy Data**\n\n"
                    "**Explanation**: The research assumes that all input data is clean and correctly formatted. However, real-world data is often corrupted by sensor noise, transmission errors, or intentional adversarial attacks.\n\n"
                    "**Why it matters**: A model that is not robust can be easily manipulated, leading to incorrect predictions that could have serious consequences in fields like healthcare or autonomous driving.\n\n"
                    "**Future Work**: Implement a noise-injection layer during the evaluation phase to measure the degradation of accuracy under various levels of data corruption.")

    if not gaps:
        gaps.append("**Gap: Cross-Domain Generalization and Adaptability**\n\n"
                    "**Explanation**: The proposed solution is highly optimized for its specific niche but lacks validation on secondary or unrelated domains. It is unclear if the underlying patterns hold true elsewhere.\n\n"
                    "**Why it matters**: High-impact research usually requires a level of 'general intelligence' or adaptability that allows the solution to solve problems beyond its initial training scope.\n\n"
                    "**Future Work**: Test the framework on at least one dataset from a completely different industry to identify core invariant features.")

    # 6. Future Directions (3-5 ideas)
    future_ideas = [
        "**Hybrid Architectures**: Combine the current approach with Transformer-based models for better context awareness.",
        "**Real-time Optimization**: Focus on reducing the inference time to allow for mobile or edge deployment.",
        "**Explainability Layer**: Add a 'Feature Importance' visualization to help users understand why the model made a specific decision.",
        "**Cross-Platform Portability**: Port the existing framework to a more lightweight library for easier integration."
    ]

    # 7. Scoring (Detailed)
    scores = {}
    justifications = {}
    
    # Clarity
    sentences = split_sentences(full_text)
    avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
    scores["Clarity"] = 2 if 80 < avg_len < 220 else 1 if avg_len > 0 else 0
    justifications["Clarity"] = "The sentence structure is well-balanced for an academic audience, ensuring that complex ideas are not lost in run-on sentences."

    # Methodology
    scores["Methodology"] = 2 if len(method) > 1500 else 1 if method else 0
    justifications["Methodology"] = "The technical framework is clearly established, though it could benefit from more granular implementation details if scored less than 2."

    # Novelty
    scores["Novelty"] = 2 if any(w in full_text for w in ["novel", "contribution", "unique"]) else 1
    justifications["Novelty"] = "The paper indicates a unique perspective or improvement over existing models, which is essential for a publishable contribution."

    # Completeness
    present_sections = sum(1 for k in ["intro", "method", "results", "conclusion"] if sections.get(k))
    scores["Completeness"] = 2 if present_sections == 4 else 1 if present_sections >= 2 else 0
    justifications["Completeness"] = "Most standard academic sections are present, providing a logical flow from hypothesis to conclusion."

    # Technical Depth
    scores["Technical Depth"] = 2 if any(kw in full_text for kw in ["complexity", "latency", "robustness"]) else 1
    justifications["Technical Depth"] = "The analysis moves beyond surface-level descriptions into deeper technical trade-offs and performance metrics."

    total_score = sum(scores.values())
    confidence = min(100, int((total_len / 20000) * 50 + (present_sections / 4) * 50))

    # Formatting Output
    output = [
        "## Paper Summary", summary_block,
        "\n## Strengths", "\n".join([f"- {s}" for s in strengths]),
        "\n## Weaknesses", "\n".join([f"- {w}" for w in weaknesses]),
        "\n## Improvement Suggestions", "\n".join([f"- {s}" for s in suggestions]),
        "\n## Research Gaps", "\n".join(gaps[:3]),
        "\n## Future Directions", "\n".join([f"- {f}" for f in future_ideas]),
        "\n## Score Explanation",
        f"### Total Score: {total_score}/10",
        f"**Confidence Score**: {confidence}% (Accuracy of heuristic mapping)",
        "\n#### Breakdown:",
    ]
    for cat in ["Clarity", "Methodology", "Novelty", "Completeness", "Technical Depth"]:
        output.append(f"- **{cat} ({scores[cat]}/2)**: {justifications[cat]}")
    
    output.append(f"\n**Verdict**: {'Ready for peer review with minor edits.' if total_score >= 8 else 'Requires significant expansion and technical rigor.'}")

    return "\n".join(output)

def detect_domain_context(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Dynamically detects the research domain and primary context from chunks."""
    text_blob = " ".join([c.get("text", "") for c in chunks]).lower()
    
    # 1. Identify Domain (Heuristic based on technical terminology)
    domain_map = {
        "Healthcare/Medical": ["health", "clinical", "medical", "patient", "ehr", "diagnosis", "doctor"],
        "Finance/Economics": ["finance", "stock", "market", "trade", "economic", "bank", "risk"],
        "Education/Learning": ["student", "tutor", "learn", "education", "teacher", "classroom", "knowledge"],
        "Media/Content Creation": ["content", "media", "generate", "creative", "video", "image", "marketing"],
        "Robotics/Autonomous": ["robot", "autonomous", "vehicle", "drone", "navigation", "sensor"],
        "Cybersecurity": ["security", "attack", "threat", "malware", "network", "encryption", "privacy"],
        "Environmental Science": ["climate", "environment", "sustainability", "energy", "carbon", "ecology"]
    }
    
    detected_domain = "General Technology and Systems Research"
    max_matches = 0
    for domain, kws in domain_map.items():
        matches = sum(1 for kw in kws if kw in text_blob)
        if matches > max_matches:
            max_matches = matches
            detected_domain = domain
            
    # 2. Extract Primary Focus (Heuristic)
    focus = "Optimizing system performance and decision-making"
    if "transformer" in text_blob or "attention" in text_blob: focus = "Large-scale sequence modeling and contextual understanding"
    elif "reinforcement" in text_blob or "policy" in text_blob: focus = "Sequential decision-making under uncertainty"
    elif "diffusion" in text_blob or "gan" in text_blob: focus = "Generative modeling and synthetic data creation"
    
    # 3. Synthesize a Real-World Example from the chunks
    example = ""
    # Try to find a sentence starting with "for example", "specifically", or describing a use case
    found_examples = re.findall(r'(?i)(?:for example|specifically|such as|in the case of|application in)([^.]{30,150}\.)', text_blob)
    if found_examples:
        example = found_examples[0].strip().capitalize()
    else:
        # Fallback to the focus if no explicit example found
        example = f"Applying {focus.lower()} to solve real-world complexities within the {detected_domain} sector."

    return {
        "domain": detected_domain,
        "focus": focus,
        "example": example
    }

def generate_topic_explanation(topic: str, chunks: List[Dict[str, Any]]) -> str:
    """Explains the topic specifically based on detected domain context."""
    topic_clean = topic.title() if topic else "Advanced Research Area"
    ctx = detect_domain_context(chunks)
    
    what = f"The research in **{topic_clean}** represents a specialized intersection of {ctx['domain']} and advanced computational modeling."
    
    explanation = [
        f"**Confidence Level**: {calculate_confidence(chunks)}",
        f"## 1. Topic Explanation",
        what,
        f"\n**Core Focus**: {ctx['focus']}",
        f"\n**Real-World Example**: {ctx['example']}",
        f"\n> **Key Insight**: The fundamental goal in this area is to leverage {ctx['domain'].split('/')[0]} data to drive {ctx['focus'].lower()} in real-world deployments."
    ]
    return "\n".join(explanation)

def generate_research_overview(chunks: List[Dict[str, Any]]) -> str:
    """Identifies real domains, provides a technical timeline, and a concise overview."""
    years = []
    for c in chunks:
        found = re.findall(r'\b(20\d{2})\b', c.get("text", ""))
        years.extend([int(y) for y in found])
    
    start_year = min(years) if years else 2017
    end_year = max(years) if years else 2024
    
    evolution = f"The field has transitioned from early rule-based heuristics (pre-{start_year+2}) to the current era of deep, data-driven architectures. "
    turning_point = "The major turning point was the widespread adoption of Transformer-based models and Large Language Models (LLMs), which allowed for handling unprecedented context and scale."
    reason = "This growth was primarily driven by the availability of high-quality, large-scale datasets and the rapid advancement in parallel computing hardware."

    return (
        "## 2. Research Overview\n"
        f"This synthesis analyzes recent peer-reviewed publications spanning from {start_year} to {end_year}. "
        f"{evolution} {turning_point} {reason}"
    )

def generate_system_flow(chunks: List[Dict[str, Any]]) -> str:
    """Generates a textual description of the system architecture."""
    flow = """## 3. System Architecture Description

The proposed architecture follows a straightforward, iterative loop designed for reliability and continuous improvement:

1. **User Input / Query**: The system captures raw interaction data or user queries.
2. **Data Processing & Cleaning**: The data is sanitized and structured for analytical processing.
3. **AI/ML Model Prediction**: The core model evaluates the structured data to generate a prediction or insight.
4. **Decision Routing (Confidence Check)**: 
   - If the model's confidence is *High*, it directly generates automated feedback or takes action.
   - If the confidence is *Low*, the system routes the query to a Human-in-the-loop or a safe, rule-based fallback.
5. **Learning Adaptation**: Both automated and human-verified outcomes are fed back into the system to continually refine and adapt the learning models for future interactions.
"""
    return flow

def generate_methodologies(chunks: List[Dict[str, Any]]) -> str:
    """Extracts actual approaches and categorizes them properly with How/Why/Example."""
    text_blob = " ".join([c.get("text", "") for c in chunks]).lower()
    approaches = []
    ctx = detect_domain_context(chunks)
    
    # 1. Rule-Based
    approaches.append("### Rule-Based Systems\n- **How**: Uses predefined 'if-then' logic and expert knowledge bases.\n- **Why**: Essential for transparency and when data is scarce.\n- **Example**: Safety constraints and boundary conditions in production systems.")
    
    # 2. ML
    if any(kw in text_blob for kw in ["machine learning", "random forest", "svm", "regression"]):
        approaches.append(f"### (⭐ Critical) Machine Learning (ML)\n- **How**: Trains statistical models on historical data to predict outcomes.\n- **Why**: Automates decision-making and identifies patterns at scale.\n- **Example**: Predictive modeling of {ctx['domain'].split('/')[0]} trends based on historical logs.\n- **Cross-link**: While robust, ML models often face the [Gap: Algorithmic Bias] identified in the Strategy section.")
        
    # 3. Deep Learning
    if any(kw in text_blob for kw in ["deep learning", "neural network", "cnn", "rnn", "transformer"]):
        approaches.append(f"### (📌 High Priority) Deep Learning (DL)\n- **How**: Utilizes multi-layered neural networks to extract high-level features.\n- **Why**: Handles complex, unstructured data like text, images, and speech.\n- **Example**: {ctx['focus']} for large-scale enterprise deployments.\n- **Cross-link**: High performance comes at the cost of [Gap: Interpretability Challenges].")
        
    # 4. RL
    if "reinforcement learning" in text_blob or "rl" in text_blob:
         approaches.append("### Reinforcement Learning (RL)\n- **How**: Uses a reward system to train agents through trial and error.\n- **Why**: Optimizes sequential decision-making in dynamic environments.\n- **Example**: Real-time policy optimization for adaptive system responses.\n- **Cross-link**: Implementation is often limited by [Gap: Data Privacy Considerations] when tracking granular state changes.")

    return "## 4. Methodologies & Approaches\n" + "\n\n".join(approaches)

def generate_key_findings(chunks: List[Dict[str, Any]]) -> str:
    """Converts raw data into polished technical insights with Real-World Impact."""
    findings = []
    seen = set()
    
    indicators = ["outperform", "achieve", "result", "demonstrate", "show", "increase"]
    
    for c in chunks:
        if any(kw in c.get("section", "").lower() for kw in ["result", "eval", "conclusion"]):
            sentences = split_sentences(c.get("text", ""))
            for s in sentences:
                if any(ind in s.lower() for ind in indicators) and len(s) > 80:
                    clean_s = s.replace("Figure", "The analysis").replace("Table", "The data")
                    if clean_s[:10] not in seen:
                        marker = "(⭐ Critical)" if "outperform" in clean_s or "achieve" in clean_s else "(📌 High Priority)"
                        impact_topic = clean_s.split(' ')[2:5] # Heuristic: grab a few words as context
                        impact_str = f"By implementing these results, practitioners can expect a measurable increase in performance specifically within the realm of {' '.join(impact_topic)}."
                        findings.append(f"### {marker} Finding: {clean_s}\n**Real-World Impact**: {impact_str}")
                        seen.add(clean_s[:10])
                        break
        if len(findings) >= 5: break
            
    if not findings:
        ctx = detect_domain_context(chunks)
        findings = [
            f"### Finding: Hybrid {ctx['domain'].split('/')[0]} models improve system consistency.\n**Real-World Impact**: Organizations can automate {ctx['focus'].lower()} with higher reliability, reducing human-in-the-loop overhead.",
            f"### Finding: Semantic alignment in models reduces uncertainty in {ctx['domain']} reporting.\n**Real-World Impact**: Decision-support tools become significantly more trustworthy for enterprise-grade deployments.",
            f"### Finding: Efficient fine-tuning allows for niche-specific personalization in {ctx['domain'].split('/')[0]}.\n**Real-World Impact**: Small-scale projects can benefit from high-performance AI architectures without massive computational costs."
        ]
        
    return "## 5. Key Findings\n" + "\n\n".join(findings)

def generate_paper_contributions(chunks: List[Dict[str, Any]]) -> str:
    """Summarizes papers using ONLY extracted data. Skips paper if info is unclear."""
    papers = defaultdict(list)
    for c in chunks:
        papers[c.get("paper_title", "Unknown")].append(c)
    
    output = ["## 5. Key Paper Contributions"]
    
    count = 0
    for title, paper_chunks in papers.items():
        if title == "Unknown" or count >= 3: continue
        
        text = " ".join([c.get("text", "") for c in paper_chunks])
        sentences = split_sentences(text)
        
        # Strictly extracted logic (Zero template)
        problem, approach, contribution = "", "", ""
        
        for s in sentences:
            if any(k in s.lower() for k in ["challenge", "problem", "motivation"]): problem = s
            elif any(k in s.lower() for k in ["propose", "method", "architecture"]): approach = s
            elif any(k in s.lower() for k in ["contribution", "results", "improvement"]): contribution = s
            if problem and approach and contribution: break
            
        if not (problem and approach and contribution):
            continue # SKIP IF UNCLEAR
        
        output.append(f"### {title}\n- **Problem**: {problem}\n- **Method**: {approach}\n- **Contribution**: {contribution}")
        count += 1
        
    if len(output) == 1: return "" # Return nothing if no clear papers found
    return "\n\n".join(output)

def generate_research_understanding() -> str:
    """Learning path guide for the topic."""
    return """## 8.5 How to Understand This Topic

To master this research area, follow this structured learning path:

1.  **Understand the Foundations**: Start by mastering the core mathematical principles (e.g., probability, linear algebra) and the basic rule-based systems that originally defined the field.
2.  **Focus on Common Approaches**: Study how standard Machine Learning models (SVMs, Random Forests) are applied to small-scale datasets before moving to deep learning.
3.  **Compare Advanced Methods**: Finally, compare state-of-the-art Transformer-based models against traditional Reinforcement Learning agents to identify which works best for specific real-world constraints.
"""

def generate_writing_assistance() -> str:
    """Structured guidance for writing a research paper."""
    return """## 11.5 How to Write About This Topic

Use this structure to build a compelling research paper:

- **Introduction**: Clearly define the topic's relevance today and mention the recent growth driven by Large Language Models (LLMs).
- **Methodology Section**: Focus on comparing different technical approaches (e.g., RL vs NLP) and justify your choice based on computational efficiency.
- **Discussion**: Address critical issues like scalability, data bias, and the 'black box' nature of deep learning models.
- **Conclusion**: Summarize the future of generative AI and its potential to revolutionize your specific domain.
"""

def generate_research_directions() -> str:
    """Innovative research ideas categorized by complexity."""
    return """## 12.5 Research Directions (Your Killer Feature)

If you are looking to start research or a project today, consider these ideas:

- **Beginner**: Build a simple classification tool that uses a pre-trained model to categorize research abstracts by their primary methodology.
- **Intermediate**: Compare the performance of an RL-based agent against a standard NLP-based system in a controlled, small-scale environment.
- **Advanced**: Design a hybrid system that combines human feedback with AI generation to ensure both creativity and factual accuracy.
"""

def generate_comparison_table(chunks: List[Dict[str, Any]]) -> str:
    """Technical comparison of methodologies."""
    lines = [
        "## 6. Comparison of Approaches", 
        "| Approach | Strength | Weakness | Best Use Case |", 
        "| :--- | :--- | :--- | :--- |",
        "| Rule-Based | Highly Explainable | Not scalable | Foundational expert systems |",
        "| Machine Learning | Adaptive & Fast | Needs structured data | Pattern recognition |",
        "| Deep Learning | High accuracy | Black box | Complex, high-dimensional data |",
        "| LLMs / NLP | Conversational | Expensive & Hallucinates | Semantic understanding |"
    ]
    return "\n".join(lines)

def generate_limitations(chunks: List[Dict[str, Any]]) -> str:
    """Detailed technical limitations, rewritten cleanly."""
    limits = [
        "- **Data Privacy Considerations**: Handling sensitive user information requires stringent anonymization to align with regulatory frameworks.",
        "- **Algorithmic Bias**: Models trained on historical datasets may inadvertently reflect existing biases in their predictions.",
        "- **Interpretability Challenges**: Deep learning models often function as 'black boxes', complicating the interpretation of their decision-making processes.",
        "- **Generalization Limitations**: Proposed models that perform well in controlled settings may require further adaptation for noisy, real-world deployment."
    ]
    return "## 7. Limitations\n" + "\n".join(limits)

def generate_experimental_setup(chunks: List[Dict[str, Any]]) -> str:
    """Dynamically identifies tools and experimental frameworks."""
    ctx = detect_domain_context(chunks)
    setup = [
        "### Evaluation Metrics",
        "- **Accuracy & F1-Score**: To measure the exactness of predictive outcomes.",
        f"- **Domain Specific Metrics**: Metrics tailored for {ctx['domain'].split('/')[0]} performance evaluation.",
        "### Tools & Frameworks",
        "- **Python**: The core language for data preprocessing and analysis.",
        f"- **Deep Learning Frameworks**: Integration of PyTorch/TensorFlow for implementing {ctx['focus'].lower()}."
    ]
    return "## 7.5 Experimental Setup\n" + "\n".join(setup)


def generate_research_gaps_agent(chunks: List[Dict[str, Any]]) -> str:
    """Identifies missing links or acknowledged limitations using dynamic extraction."""
    text_blob = " ".join([c.get("text", "") for c in chunks])
    sentences = split_sentences(text_blob)
    gaps = []
    
    # Heuristic for gaps: "future work", "limitation", "lack of", "remains a challenge"
    for s in sentences:
        if any(k in s.lower() for k in ["future work", "limitation", "lack of", "challenge", "not yet", "insufficient"]):
            reason = "unexplored" if "not yet" in s.lower() else "technical"
            gaps.append(f"- {s}\n  **Why it matters**: This {reason} bottleneck prevents existing systems from achieving full autonomy in production environments.")
        if len(gaps) >= 4: break
        
    if not gaps:
        gaps = [
            "- **Real-time Scalability**: Most current models face latency issues when deployed in live, high-traffic environments.",
            "- **Interpretability**: The 'black box' nature of deep architectures remains a major barrier for adoption in high-stakes industries.",
            "- **Domain Generalization**: Models trained on specific datasets often fail when applied to slightly different real-world distributions."
        ]
        
    return "## 8. Research Gaps & Limitations\n" + "\n".join(gaps)

def generate_future_ideas_agent(chunks: List[Dict[str, Any]]) -> str:
    """Meaningful and innovative research directions."""
    ctx = detect_domain_context(chunks)
    ideas = [
        f"- **Real-time Adaptive Systems in {ctx['domain'].split('/')[0]}**: Leveraging low-latency models to dynamically adjust to user input on the fly.",
        f"- **Context-Aware {ctx['focus']}**: Integrating multi-modal analysis to gauge user engagement and adapt system behavior accordingly.",
        f"- **Personalized Data Pipelines**: Using Reinforcement Learning (RL) to continuously optimize the sequence of actions delivered to the user."
    ]
    return "## 9. Future Research Work\n" + "\n".join(ideas)

def generate_use_case(topic: str) -> str:
    return "## 10. Real-World Use Case\n> **Target Scenario**: An AI Assistant for specialized environments (e.g., tailored AI Tutors for competitive exam students like UPSC/JEE, or specialized Clinical Support AI).\n\nThis application bridges the gap between raw data analysis and actionable insights, providing immediate value to end-users."

def generate_implementation(topic: str) -> str:
    return "## 11. Implementation Idea\nTo build this system in a production environment:\n- **Frontend**: Python + Streamlit (for rapid prototyping) or React/Next.js for a robust web app.\n- **Backend Engine**: Flask or FastAPI connecting to the core logic.\n- **AI Layer**: Integration with LLM APIs (e.g., OpenAI, Gemini, Claude) for synthesis and NLP tasks.\n- **Database**: Vector DB (like FAISS or Pinecone) for semantic retrieval."

def generate_contribution(topic: str) -> str:
    return "## 12. Core Contribution\n> **This report contributes by:**\n\n- Synthesizing AI, ML, and NLP methodologies into a unified analytical framework.\n- Identifying the critical trust and explainability gaps in current systems.\n- Proposing a human-centric, adaptable architecture tailored for real-world deployment."

def generate_system_design(chunks: List[Dict[str, Any]]) -> str:
    """Practical reasoning and architecture proposal."""
    ctx = detect_domain_context(chunks)
    design = [
        f"If tasked with designing this system for a production environment in the {ctx['domain']} sector, I would prioritize a hybrid, decoupled architecture:",
        f"1. **Core Processing Layer**: Utilize a modular architecture based on {ctx['focus'].lower()} to accurately capture complex temporal dependencies in real-time data streams.",
        "2. **Policy Optimization Layer**: Implement an offline Reinforcement Learning (RL) agent that continually updates personalized pathways without requiring risky, real-time exploration on active production environments.",
        "3. **Interaction Layer**: Deploy a specialized Large Language Model fine-tuned specifically on domain-relevant data, heavily constrained by rule-based guardrails to prevent hallucination.",
        "4. **Trade-offs**: While this multi-layered approach increases computational overhead compared to monolithic systems, the modularity ensures that high-risk generative components are securely isolated from the core logic engine."
    ]
    return "## 9.5 If I Were Designing This System\n" + "\n\n".join(design)

def generate_references(chunks: List[Dict[str, Any]]) -> str:
    """Consolidated references list based on extracted paper data."""
    papers = set()
    refs = []
    for c in chunks:
        title = c.get("paper_title", "Unknown")
        link = c.get("link", "#")
        if title != "Unknown" and title not in papers:
            refs.append(f"- **{title}**: [Link]({link})")
            papers.add(title)
            
    if not refs:
        refs = ["- No specific references found in the filtered segments."]
        
    return "## 13. References\n" + "\n".join(refs[:10])

def generate_query_plan(topic: str, api_key: str) -> List[str]:
    """Generates 3 or 5 sub-research questions based on topic complexity."""
    if not genai: return [topic]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""Analyze the research topic: "{topic}"
    
    1. Determine if the topic is 'Simple' (narrow scope) or 'Complex' (broad/interdisciplinary).
    2. If Simple, generate 3 specific research sub-questions.
    3. If Complex, generate 5 specific research sub-questions.
    
    Format the output ONLY as a bulleted list of questions. Do not include any other text.
    """
    try:
        response = model.generate_content(prompt)
        questions = re.findall(r'^\s*[-*•]\s*(.*)$', response.text, re.MULTILINE)
        return [q.strip() for q in questions] if questions else [topic]
    except Exception as e:
        print(f"Query Plan Error: {e}")
        return [topic]

def detect_contradictions(chunks: List[Dict[str, Any]], api_key: str) -> str:
    """Identifies disagreements, trade-offs, or debates between papers."""
    if not genai or not chunks: return ""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    context = "\n".join([f"- [{c.get('paper_title')}] {c.get('text')}" for c in chunks[:15]])
    
    prompt = f"""Compare the following research segments.
    Identify any:
    - Contradictions in findings or results.
    - Differing methodologies and why one might be preferred over another.
    - Trade-offs mentioned (e.g., accuracy vs. speed).
    - Scholarly debates or unresolved questions.
    
    Context:
    {context}
    
    Format as a structured section with headers like '### Contradictions' or '### Trade-offs'. 
    If no major tensions are found, return "No significant contradictions detected between current sources."
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Tension Detection Error: {e}")
        return ""

def evaluate_report(report: str, api_key: str) -> Dict[str, Any]:
    """Self-reflection: evaluates the report for completeness and quality."""
    if not genai: return {"quality_score": 10, "weak_sections": []}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""Act as a critical peer reviewer. Evaluate the following research report:
    
    REPORT:
    {report}
    
    CRITERIA:
    1. Does it sound like a generic AI response or a deep technical synthesis?
    2. Are the 'Research Gaps' specific and meaningful?
    3. Is there a clear logical flow?
    4. Identify any "weak sections" that lack technical depth or use too many placeholders.
    
    Output JSON format:
    {{
        "quality_score": (1-10),
        "weak_sections": ["Section Header 1", "Section Header 2"],
        "critical_feedback": "Short summary of what to improve"
    }}
    """
    try:
        response = model.generate_content(prompt)
        import json
        # Extract JSON from markdown if needed
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"quality_score": 7, "weak_sections": []}
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return {"quality_score": 7, "weak_sections": []}

def refine_report(report: str, feedback: Dict[str, Any], api_key: str) -> str:
    """Post-reflection refinement pass to improve weak sections."""
    if not genai or feedback.get("quality_score", 10) >= 9 or not feedback.get("weak_sections"):
        return report
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""Refine the following research report based on critical peer review feedback.
    
    FEEDBACK:
    {feedback.get('critical_feedback')}
    WEAK SECTIONS TO REWRITE: {", ".join(feedback.get('weak_sections', []))}
    
    STRICT RULES:
    - Rewrite the weak sections to be more technically dense and specific.
    - Remove any generic "placeholder" language.
    - Maintain the original report's structure but improve the depth of reasoning.
    
    ORIGINAL REPORT:
    {report}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Refinement Error: {e}")
        return report

def generate_report_with_llm(chunks: List[Dict[str, Any]], topic: str, api_key: str, themed_context: str = "", tensions: str = "", mode: str = "Student") -> str:
    """Uses Google Gemini API to generate a clean, expert-level academic report with full synthesis."""
    if not genai:
        return None
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # Strictly use top 10 best chunks
    context = themed_context if themed_context else "\n".join([f"- {c['text']}" for c in chunks[:10]])
    
    mode_instructions = ""
    if mode.lower() == "student":
        mode_instructions = "- Provide beginner-friendly explanations and simplify technical concepts.\n- Include writing guidance and a learning roadmap.\n- Avoid overly dense technical jargon."
    elif mode.lower() == "researcher":
        mode_instructions = "- Provide deeper technical analysis and advanced methodologies.\n- Highlight contradictions and debates between papers.\n- Identify critical research gaps and future directions.\n- Use strong technical vocabulary."
    else:
        mode_instructions = "- Provide a concise and brief summary."

    prompt = f"""You are an expert academic researcher.

You are NOT allowed to copy or reuse any raw text from the input research context.

Your job is to:
- understand the research context provided below
- rewrite everything in your own words
- explain clearly and accurately for a {mode.lower()} audience

STRICT RULES:
{mode_instructions}

1. If input text is:
   - broken (e.g. contains "rithms", "onebox", etc.)
   - incomplete
   - unclear
→ IGNORE IT completely. 

2. Every output sentence must be:
   - clean
   - meaningful
   - human-readable

3. Paper Contributions:
   - MUST be unique per paper.
   - MUST include: Problem, Method, and Actual Contribution (be specific).
   - If paper info is unclear or fragmented → DO NOT INCLUDE IT.

4. Research Gaps:
   - MUST be explained in simple language.
   - MUST include: what is missing, why it matters, and a future direction.

5. Topic Alignment:
   - ONLY include content directly related to the TOPIC below.
   - Strictly REMOVE unrelated domains (e.g. no "Education" or "Clinical" if topic is "Content Creation").

6. Real-world examples:
   - Must be specific, practical, and meaningful.

7. NEVER generate generic placeholder statements like:
   - "technical advancement"
   - "empirical validation"
   - "significant improvement"

TOPIC:
{topic}

---

RESEARCH CONTEXT:
{context}

---

RESEARCH TENSIONS & TRADE-OFFS:
{tensions if tensions else "Analyze the context for inherent trade-offs and debates."}

---

Goal:
Write a clean, intelligent research report that feels like it was written by a human expert—not generated from raw data. 
Group the output into these block markers: ### [FOUNDATION], ### [TECHNICAL], and ### [STRATEGY].
Include a specific subsection in ### [TECHNICAL] for 'Contradictions and Trade-offs' based on the provided tensions.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def format_segments(chunks: List[Dict[str, Any]], topic: str = "Research Topic", mode: str = "student", angle: str = "methods", gemini_api_key: str = None, themed_context: str = "", tensions: str = "") -> str:
    """Consolidates all segments into a final report, using Gemini if available."""
    if gemini_api_key:
        report = generate_report_with_llm(chunks, topic, gemini_api_key, themed_context=themed_context, tensions=tensions, mode=mode)
        if report:
            refs = generate_references(chunks)
            return report + "\n\n### [REFERENCES]\n" + refs

    # HEURISTIC FALLBACK (Structured into 3 main visual blocks for Streamlit UI)
    foundation = [
        "### [FOUNDATION]",
        generate_topic_explanation(topic, chunks),
        generate_research_overview(chunks),
    ]
    
    technical = [
        "### [TECHNICAL]",
        generate_methodologies(chunks),
        generate_key_findings(chunks),
        generate_paper_contributions(chunks)
    ]
    
    strategy = [
        "### [STRATEGY]",
        generate_research_gaps_agent(chunks),
        generate_future_ideas_agent(chunks),
        generate_system_design(chunks),
        generate_use_case(topic)
    ]
    
    references = [
        "### [REFERENCES]",
        generate_references(chunks)
    ]
    
    full_report = [
        "\n\n".join(foundation),
        "\n\n".join(technical),
        "\n\n".join(strategy),
        "\n\n".join(references)
    ]
    
    return "\n\n---\n\n".join(full_report)

def chat_with_report(query: str, history: List[Dict[str, str]], report: str, api_key: str) -> str:
    """Answers follow-up questions using the generated report as strict context to prevent hallucination."""
    if not genai:
        return "Gemini API is not available."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # Format chat history
    history_text = ""
    for msg in history[-5:]: # Keep last 5 messages for context window management
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
        
    prompt = f"""You are an interactive AI research mentor. 

Your task is to answer the user's latest question based STRICTLY on the provided Research Report and Conversation History.

STRICT RULES:
1. Prevent Hallucinations: If the answer cannot be found in the report or is completely unrelated to the research domain, politely state that you can only answer questions based on the retrieved papers.
2. Beginner-Friendly: Explain technical concepts simply. Do not use excessive jargon.
3. Tone: Helpful, mentoring, and academic.
4. Keep it concise.

RESEARCH REPORT (Ground Truth):
{report}

---
CONVERSATION HISTORY:
{history_text}

---
USER'S LATEST QUESTION:
{query}

ANSWER:
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Chat Error: {e}")
        return "I encountered an error while generating a response. Please try again."
