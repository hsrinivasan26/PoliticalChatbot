from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field #output schema
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, TAVILY_API_KEY, OPENAI_MODEL
from prompts import RESEARCH_SYNTHESIS_SYSTEM_PROMPT, CONSTITUTIONAL_SYNTHESIS_SYSTEM_PROMPT
from rag import get_index

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
)

search_tool = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=10,
    exclude_domains=["wikipedia.org", "reddit.com", "quora.com", "twitter.com", "x.com", "facebook.com", "tiktok.com", "pinterest.com", "instagram.com"],
)

#FAISS stores index of constitution sections; see rag.py
_constitution_index = get_index()

#good argumentative reasoning follows claim-evidence-reasoning structure. Chatbot should
#never make a claim without evidence, and reasoning enforces binding the evidence to the claim so that 
#the chatbot never brings up random info to "justify" what it's saying
class Claim(BaseModel):
    proposition: str
    evidence: list[tuple[str, str]]
    reasoning: str

#Evidence used should map one-to-one to source links. No unsourced claims allowed.
class _EvidencePair(BaseModel):
    quote: str = Field(description="A direct quotation from the source material.")
    source: str = Field(description="The link to the source, or an amendment/article number if constitutional.")

#We need this bc a list of tuples gets rejected by OpenAI structured outputs, it can't be produced
#This intermediate class allows the structured output to give a list of EvidencePair objects instead
#and then post-facto converts them into string pairs to be processed by the final chat LLM
class _ClaimSchema(BaseModel):
    proposition: str = Field(description="The claim being made, answering the query.")
    evidence: list[_EvidencePair] = Field(default_factory=list, description="Quotes paired with their sources supporting the proposition.")
    reasoning: str = Field(description="How the evidence supports the proposition.")

claim_synthesizer = llm.with_structured_output(_ClaimSchema)

#no evidence field here on purpose baconstitutional evidence is pulled verbatim from the RAG
#final Chat LLM extracts relevant quotes from the articles/amendments sourced from RAG
class _ConstitutionalClaimSchema(BaseModel):
    proposition: str = Field(description="The claim being made, answering the query.")
    reasoning: str = Field(description="How the evidence supports the proposition.")

constitutional_synthesizer = llm.with_structured_output(_ConstitutionalClaimSchema)


def _format_search_results(results):
    lines = []
    for i, r in enumerate(results.get("results", [])):
        lines.append(f"{i + 1}. Title: {r.get('title')}\n   URL: {r.get('url')}\n   Content: {r.get('content')}")
    return "\n".join(lines)

#convertes ClaimSchema objects to the nice Claim objects
def _synthesize_claim(query, results):
    raw_claim = claim_synthesizer.invoke([
        SystemMessage(content=RESEARCH_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}\n\nSearch results:\n{_format_search_results(results)}"),
    ])
    return Claim(
        proposition=raw_claim.proposition,
        evidence=[(e.quote, e.source) for e in raw_claim.evidence],
        reasoning=raw_claim.reasoning,
    )

def _synthesize_constitutional_claim(query, docs):
    excerpts = "\n".join(f"{i + 1}. [{doc.metadata['reference']}] {doc.page_content}" for i, doc in enumerate(docs))
    raw_claim = constitutional_synthesizer.invoke([
        SystemMessage(content=CONSTITUTIONAL_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}\n\nExcerpts:\n{excerpts}"),
    ])
    return Claim(
        proposition=raw_claim.proposition,
        evidence=[(doc.page_content, doc.metadata["reference"]) for doc in docs],
        reasoning=raw_claim.reasoning,
    )

def _run_factual(query):
    results = search_tool.invoke(query)
    return _synthesize_claim(query, results)

def _run_perspective(pair):
    perspective_query, entity = pair
    query = f"{entity} {perspective_query}"
    results = search_tool.invoke(query)
    return _synthesize_claim(query, results)

def _run_constitutional(query):
    docs = _constitution_index.similarity_search(query, k=2)
    return _synthesize_constitutional_claim(query, docs)

#final output for the research section. Maps queries to researched claims, to be synthesized
#by the core chat interface. Every inquiry is independent of every other, so they all run
#concurrently in a thread pool scoped to this call--search/LLM calls are I/O-bound, so threads
#are enough here without needing to rewrite everything as async
def research_output(inquiry):
    tasks = (
        [(_run_factual, q) for q in inquiry.factual_inquiries]
        + [(_run_perspective, p) for p in inquiry.perspective_inquiries]
        + [(_run_constitutional, q) for q in inquiry.constitutional_inquiries]
    )
    if not tasks:
        return []

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(fn, arg) for fn, arg in tasks]
        claims = [f.result() for f in futures]

    return claims

def set_model(model_name):
    global llm, claim_synthesizer, constitutional_synthesizer
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    claim_synthesizer = llm.with_structured_output(_ClaimSchema)
    constitutional_synthesizer = llm.with_structured_output(_ConstitutionalClaimSchema)
