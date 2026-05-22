import logging
import sys

# Ensure UTF-8 for better Windows compatibility
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from src.agent import answer_follow_up, run_agent

# Load environment variables
load_dotenv()

# Configure logging to stderr so Claude Desktop can capture it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP Server
mcp = FastMCP("Research Agent Server")

@mcp.tool()
def research(topic: str) -> str:
    """
    Search for research papers on a specific topic.
    Returns structured data segments from relevant papers.
    """
    logger.info("Tool Call: research(topic=%r)", topic)
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        result = run_agent(topic, gemini_api_key=gemini_api_key)
        logger.info("research(topic=%r) succeeded.", topic)
        return result
    except Exception as e:
        logger.exception("Error in research(topic=%r): %s", topic, e)
        return f"Error executing research: {str(e)}"

@mcp.tool(name="ask-agent")
def ask_agent(query: str) -> str:
    """
    Ask a follow-up question based on previously indexed research papers.
    Returns relevant segments of evidence.
    """
    logger.info("Tool Call: ask-agent(query=%r)", query)
    try:
        result = answer_follow_up(query)
        logger.info("ask-agent(query=%r) succeeded.", query)
        return result
    except Exception as e:
        logger.exception("Error in ask-agent(query=%r): %s", query, e)
        return f"Error executing follow-up: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting Research Agent MCP Server...")
    mcp.run()
