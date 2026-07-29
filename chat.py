from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import OPENAI_API_KEY, TAVILY_API_KEY, OPENAI_MODEL

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
)

search_tool = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    max_results=20,
)
!
