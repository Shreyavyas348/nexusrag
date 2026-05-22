import streamlit as st
import os
import time
import base64
import logging
import re
from dotenv import load_dotenv
from src.agent import run_agent
from src.pdf_generator import generate_report_pdf
from src.llm import analyze_user_paper, chat_with_report
from src.pdf_reader import extract_raw_text, extract_structured_sections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="NexusRAG | Intelligent Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-Safe CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }

    /* Section Containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 2rem;
    }

    /* Source Card Styling */
    .source-card {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #7aa2f7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: inherit;
    }

    .source-title {
        color: #7aa2f7;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    /* Highlight Boxes */
    .highlight-info {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 1.25rem;
        border-radius: 0 8px 8px 0;
        color: #0369A1;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .highlight-warning {
        background-color: #FFFbeb;
        border-left: 4px solid #F59E0B;
        padding: 1.25rem;
        border-radius: 0 8px 8px 0;
        color: #B45309;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .highlight-success {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 1.25rem;
        border-radius: 0 8px 8px 0;
        color: #047857;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Customizing Streamlit Elements */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
    }
    .stSelectbox > div > div > div {
        border-radius: 8px;
    }
    
    /* Action Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* PDF Viewer Container */
    .pdf-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .main-header { color: #F8FAFC; }
        .sub-header { color: #94A3B8; }
        .glass-container { background: rgba(30, 41, 59, 0.5); border-color: rgba(255,255,255,0.05); }
        .highlight-info { background-color: rgba(14, 165, 233, 0.1); color: #7DD3FC; border-color: #0EA5E9; }
        .highlight-warning { background-color: rgba(245, 158, 11, 0.1); color: #FCD34D; border-color: #F59E0B; }
        .highlight-success { background-color: rgba(16, 185, 129, 0.1); color: #6EE7B7; border-color: #10B981; }
    }
</style>
""", unsafe_allow_html=True)

def display_pdf(file_path):
    """Embeds PDF in an iframe using base64 encoding."""
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 5 * 1024 * 1024:  # 5MB Limit
            st.warning("PDF file too large for preview (>5MB). Please use the download button below.")
            return

        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf" class="pdf-container"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering PDF preview: {e}")

def parse_report_sections(report_text):
    """Extracts sections based on major blocks: FOUNDATION, TECHNICAL, STRATEGY, REFERENCES."""
    sections = {}
    blocks = ["FOUNDATION", "TECHNICAL", "STRATEGY", "REFERENCES"]
    
    for block in blocks:
        # Flexible regex to find the block header.
        pattern = re.compile(rf"(?:#+\s*\[?{block}\]?|\*\*\[?{block}\]?\*\*|\[{block}\])", re.IGNORECASE)
        match = pattern.search(report_text)
        
        if match:
            start_idx = match.start()
            
            # Find the start of the next block
            next_start = -1
            for next_block in blocks:
                if next_block == block: continue
                next_pattern = re.compile(rf"(?:#+\s*\[?{next_block}\]?|\*\*\[?{next_block}\]?\*\*|\[{next_block}\])", re.IGNORECASE)
                next_match = next_pattern.search(report_text[start_idx + len(match.group()):])
                if next_match:
                    idx = start_idx + len(match.group()) + next_match.start()
                    if next_start == -1 or idx < next_start:
                        next_start = idx
            
            if next_start != -1:
                content = report_text[start_idx:next_start].strip()
            else:
                content = report_text[start_idx:].strip()
            
            sections[block] = content
            
    return sections

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=80)
    st.title("NexusRAG Assistant")
    
    app_feature = st.radio(
        "Select Feature",
        ["Research Agent", "Paper Review"],
        index=0,
        help="Research Agent: Search & synthesize papers. Paper Review: Critique your own PDF."
    )
    
    st.divider()

    if app_feature == "Research Agent":
        st.markdown("**Search Settings**")
        mode = st.selectbox(
            "Synthesis Mode",
            ["Student", "Researcher", "Quick"],
            index=0,
            help="Student: Actionable roadmap. Researcher: Gaps & ideas. Quick: Brief summary."
        )
        angle = st.selectbox(
            "Research Focus",
            ["Methods", "Applications", "Limitations", "Scalability", "Human-Centric"],
            index=0,
            help="Prioritizes information based on this specific perspective."
        )
        paper_limit = st.slider("Paper Search Limit", 5, 20, 10)
    else:
        st.markdown("**Review Settings**")
        st.caption("Upload your draft to receive a structural critique and improvement roadmap.")

    st.divider()
    st.info("🧬 **NexusRAG** uses FAISS indexing and heuristic-based synthesis to provide factual results without LLM hallucinations.")

# Feature: Research Agent
if app_feature == "Research Agent":
    st.markdown('<div class="main-header">🧬 NexusRAG Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transforming academic segments into actionable knowledge.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Configure Analysis</div>', unsafe_allow_html=True)

    # Search Area
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input("Research Topic:", placeholder="e.g. Transformers in Vision Tasks", label_visibility="collapsed")
    with col2:
        search_clicked = st.button("Generate Report", use_container_width=True, type="primary")

    if search_clicked:
        if not topic.strip():
            st.error("⚠️ Please enter a research topic to proceed.")
        else:
            try:
                with st.status("🔍 Initializing research pipeline...", expanded=True) as status:
                    st.write("📡 Searching academic databases...")
                    start_time = time.time()
                    
                    gemini_api_key = os.getenv("GEMINI_API_KEY")
                    semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                    
                    report = run_agent(topic, limit=paper_limit, mode=mode, angle=angle, gemini_api_key=gemini_api_key, semantic_scholar_key=semantic_scholar_key)
                    st.write("📝 Synthesizing findings...")
                    st.session_state.report = report
                    st.session_state.last_topic = topic
                    st.session_state.chat_history = [] # Reset chat history
                    end_time = time.time()
                    status.update(label=f"✅ Report Generated in {end_time - start_time:.1f}s", state="complete", expanded=False)
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                logger.exception("Agent execution failed")

    st.markdown('</div>', unsafe_allow_html=True)
    
# Feature: Paper Review
else:
    st.markdown('<div class="main-header">📄 Research Paper Reviewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Heuristic-based structural critique of your research draft.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Run Structural Analysis", type="primary"):
            try:
                with st.spinner("Analyzing document structure..."):
                    pdf_bytes = uploaded_file.read()
                    text = extract_raw_text(pdf_bytes)
                    if not text.strip():
                        st.error("Could not extract text from the PDF. It might be image-based or encrypted.")
                    else:
                        sections = extract_structured_sections(text)
                        review_report = analyze_user_paper(sections)
                        st.session_state.review_report = review_report
                        st.session_state.review_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.exception("Review failed")
                
    st.markdown('</div>', unsafe_allow_html=True)

# Display Agent Results
if app_feature == "Research Agent" and "report" in st.session_state:
    sections = parse_report_sections(st.session_state.report)
    st.divider()
    
    head_col, dl_col = st.columns([3, 1])
    with head_col:
        st.subheader(f"Analysis: {st.session_state.last_topic}")
    with dl_col:
        try:
            pdf_path = f"generated_reports/report_{int(time.time())}.pdf"
            generate_report_pdf(st.session_state.report, pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF", data=f, file_name=f"Research_Report_{st.session_state.last_topic.replace(' ', '_')}.pdf", mime="application/pdf", use_container_width=True)
            st.session_state.pdf_path = pdf_path
        except: pass

    # Confidence Indicator
    full_text = st.session_state.report
    conf_level = "Medium"
    if "Confidence Level: High" in full_text or "**High**" in full_text: conf_level = "High"
    elif "Confidence Level: Low" in full_text or "**Low**" in full_text: conf_level = "Low"
    
    conf_colors = {"High": "lightgreen", "Medium": "orange", "Low": "red"}
    st.markdown(f"**AI Confidence Level**: <span style='color:{conf_colors.get(conf_level)}; font-weight:bold;'>{conf_level}</span>", unsafe_allow_html=True)

    rtab1, rtab2, rtab3, rtab4 = st.tabs(["📖 Foundation", "💡 Technical Depth", "🚀 Strategy", "📑 References"])
    
    with rtab1:
        if "FOUNDATION" in sections:
            st.markdown(sections["FOUNDATION"])
        else:
            st.info("Foundation data is being synthesized...")
    
    with rtab2:
        if "TECHNICAL" in sections:
            st.markdown(sections["TECHNICAL"])
        else:
            st.info("Technical analysis is being synthesized...")
                
    with rtab3:
        if "STRATEGY" in sections:
            st.markdown(sections["STRATEGY"])
        else:
            st.info("Strategy and roadmap are being synthesized...")
                
    with rtab4:
        if "REFERENCES" in sections:
            st.markdown(sections["REFERENCES"])
        else:
            # Fallback to old header if not found in blocks
            for h, content in sections.items():
                if "References" in h:
                    st.markdown(content)
                    
    # Interactive Chat UI
    st.divider()
    st.subheader("💬 Research Mentor Chat")
    st.markdown("Ask follow-up questions about the generated report.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if query := st.chat_input("Ask a question about the report..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing report..."):
                gemini_key = os.getenv("GEMINI_API_KEY")
                response = chat_with_report(query, st.session_state.chat_history[:-1], st.session_state.report, gemini_key)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display Review Results
elif app_feature == "Paper Review" and "review_report" in st.session_state:
    st.divider()
    head_col, dl_col = st.columns([3, 1])
    with head_col:
        st.subheader(f"Review: {st.session_state.review_filename}")
    with dl_col:
        pdf_path = f"generated_reports/review_{int(time.time())}.pdf"
        generate_report_pdf(st.session_state.review_report, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Review", data=f, file_name=f"Review_{st.session_state.review_filename}", mime="application/pdf", use_container_width=True)

    report_text = st.session_state.review_report
    if "Insufficient Content" in report_text or "Analysis Blocked" in report_text:
        st.warning(report_text)
    else:
        rtab1, rtab2, rtab3 = st.tabs(["🔍 Paper Analysis", "🚀 Improvement Roadmap", "📊 Scoring System"])
        
        review_parts = report_text.split("\n## ")
        with rtab1:
            for part in review_parts:
                if part.startswith("Paper Summary"): st.markdown("## " + part)
                if part.startswith("Strengths"): st.markdown("## " + part)
                if part.startswith("Weaknesses"): st.markdown("## " + part)
                
        with rtab2:
            for part in review_parts:
                if part.startswith("Improvement Suggestions"): st.markdown("## " + part)
                if part.startswith("Research Gaps"): st.markdown("## " + part)
                if part.startswith("Future Directions"): st.markdown("## " + part)
                
        with rtab3:
            for part in review_parts:
                if part.startswith("Score Explanation"): st.markdown("## " + part)
            st.caption("Score and Confidence are calculated via advanced technical heuristics.")

elif app_feature == "Research Agent" and "report" not in st.session_state:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 1. Fetch")
        st.caption("We scan Semantic Scholar and ArXiv.")
    with col2:
        st.markdown("### 🏗️ 2. Index")
        st.caption("Content is indexed for precise retrieval.")
    with col3:
        st.markdown("### ✍️ 3. Synthesize")
        st.caption("Agent builds a structured report.")
    st.info("👆 Enter a topic above to start.")

# Cleanup handling
