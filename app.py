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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Reduce empty spacing in Streamlit layout */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1300px;
    }
    
    /* Reduce vertical gaps between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    .stVerticalBlock {
        gap: 0.5rem !important;
    }

    /* Header Styling */
    .main-header {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
        margin-top: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: -0.01em;
    }

    /* Section Containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1.5rem 2rem !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 1.5rem !important;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Align column contents bottom-wise for input + button alignment */
    .glass-container div[data-testid="column"] {
        justify-content: flex-end !important;
    }

    /* Source Card Styling */
    .source-card {
        background: rgba(30, 41, 59, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid #6366f1 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .source-card:hover {
        border-left-color: #818cf8 !important;
        background: rgba(30, 41, 59, 0.5) !important;
        transform: translateX(3px);
    }

    .source-title {
        color: #a5b4fc !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Highlight Boxes */
    .highlight-info {
        background-color: rgba(14, 165, 233, 0.1) !important;
        border-left: 4px solid #0ea5e9 !important;
        padding: 1.25rem !important;
        border-radius: 8px !important;
        color: #7dd3fc !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.05) !important;
    }
    
    .highlight-warning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #f59e0b !important;
        padding: 1.25rem !important;
        border-radius: 8px !important;
        color: #fcd34d !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.05) !important;
    }
    
    .highlight-success {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid #10b981 !important;
        padding: 1.25rem !important;
        border-radius: 8px !important;
        color: #6ee7b7 !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.05) !important;
    }

    /* Customizing Streamlit Elements */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #f8fafc !important;
        padding: 0.75rem 1rem !important;
        height: 46px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
    }
    .stSelectbox > div > div > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    
    /* Action Buttons (Primary / Secondary) */
    .stButton > button[kind="primary"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: 46px !important;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0px 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }
    
    .stButton > button[kind="secondary"], .stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        height: 42px !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: #f8fafc !important;
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        background-color: rgba(30, 41, 59, 0.2) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1 !important;
        background-color: rgba(30, 41, 59, 0.45) !important;
    }

    /* PDF Viewer Container */
    .pdf-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Intro Card Design */
    .step-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.75rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .step-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
    }
    .step-icon {
        font-size: 2.25rem;
        margin-bottom: 0.75rem;
    }
    .step-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    .step-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
        padding: 10px 16px !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #a5b4fc !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* Chat message container styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stChatMessageUser"] {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Make chat input float beautifully */
    [data-testid="stChatInput"] {
        background-color: #131b2e !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }

    /* Sidebar Container styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding-top: 1.5rem !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.01em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    /* Logo styling in sidebar */
    [data-testid="stSidebar"] img {
        margin-bottom: 0rem !important;
        filter: drop-shadow(0px 4px 10px rgba(99, 102, 241, 0.3));
    }
    
    /* Spacing between inputs in sidebar */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Radio and widget selectors styling inside sidebar */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #a5b4fc !important;
    }

    .stSelectbox label, .stSlider label {
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #cbd5e1 !important;
        margin-bottom: 0.25rem !important;
    }
    
    hr {
        margin: 1.25rem 0 !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
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
    
    conf_colors = {
        "High": {"bg": "rgba(16, 185, 129, 0.15)", "text": "#34d399", "border": "rgba(16, 185, 129, 0.3)"},
        "Medium": {"bg": "rgba(245, 158, 11, 0.15)", "text": "#fbbf24", "border": "rgba(245, 158, 11, 0.3)"},
        "Low": {"bg": "rgba(239, 68, 68, 0.15)", "text": "#f87171", "border": "rgba(239, 68, 68, 0.3)"}
    }
    cfg = conf_colors.get(conf_level, conf_colors["Medium"])
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem; margin-top: 0.5rem;">'
        f'<span style="font-size: 0.95rem; font-weight: 600; color: #94a3b8;">AI Confidence Level:</span>'
        f'<span style="background-color: {cfg["bg"]}; color: {cfg["text"]}; border: 1px solid {cfg["border"]}; '
        f'padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;">{conf_level}</span>'
        f'</div>', 
        unsafe_allow_html=True
    )

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
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-icon">🔍</div>'
            '<div class="step-title">1. Fetch</div>'
            '<div class="step-description">Autonomous deep scan of academic databases including Semantic Scholar and arXiv to gather targeted peer-reviewed documents.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-icon">🏗️</div>'
            '<div class="step-title">2. Index</div>'
            '<div class="step-description">Chunking and vector indexing of academic papers using FAISS to support highly specific, context-rich retrieval.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-icon">✍️</div>'
            '<div class="step-title">3. Synthesize</div>'
            '<div class="step-description">Leverages advanced RAG pipelines to generate comprehensive, structured research reviews without hallucinations.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight-info" style="text-align: center; border-left: none; padding: 1.25rem 1rem; font-size: 0.95rem;">'
        '🧬 <b>Get Started:</b> Type a research topic in the search bar above and click "Generate Report" to begin.'
        '</div>',
        unsafe_allow_html=True
    )

# Cleanup handling
