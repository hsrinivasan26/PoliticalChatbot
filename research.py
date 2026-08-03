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
constitution_index = get_index()

#good argumentative reasoning follows claim-evidence-reasoning structure. Chatbot should
#never make a claim without evidence, and reasoning enforces binding the evidence to the claim so that 
#the chatbot never brings up random info to "justify" what it's saying
class Claim(BaseModel):
    proposition: str
    evidence: list[tuple[str, str]]
    reasoning: str

#Evidence used should map one-to-one to source links. No unsourced claims allowed.
class EvidencePair(BaseModel):
    quote: str = Field(description="A direct quotation from the source material.")
    source: str = Field(description="The link to the source, or an amendment/article number if constitutional.")

#We need this bc a list of tuples gets rejected by OpenAI structured outputs, it can't be produced
#This intermediate class allows the structured output to give a list of EvidencePair objects instead
#and then post-facto converts them into string pairs to be processed by the final chat LLM
class ClaimSchema(BaseModel):
    proposition: str = Field(description="The claim being made, answering the query.")
    evidence: list[EvidencePair] = Field(default_factory=list, description="Quotes paired with their sources supporting the proposition.")
    reasoning: str = Field(description="How the evidence supports the proposition.")

raw_claims = llm.with_structured_output(ClaimSchema)

#no evidence field here on purpose baconstitutional evidence is pulled verbatim from the RAG
#final Chat LLM extracts relevant quotes from the articles/amendments sourced from RAG
class ConstitutionalSchema(BaseModel):
    proposition: str = Field(description="The claim being made, answering the query.")
    reasoning: str = Field(description="How the evidence supports the proposition.")

raw_constitutional = llm.with_structured_output(ConstitutionalSchema)

#OpenAI's content filter often rejects structured-output completions that include copyrighted text
#this is non deterministic so we try a couple times
RETRY_LIMIT = 5
FALLBACK_TEXT = "This request returned an API error. You may want to try again."


def format_search_results(results):
    lines = []
    for i, r in enumerate(results.get("results", [])):
        lines.append(f"{i + 1}. Title: {r.get('title')}\n   URL: {r.get('url')}\n   Content: {r.get('content')}")
    return "\n".join(lines)

#convertes ClaimSchema objects to the nice Claim objects
def claim_builder(query, results):
    for attempt in range(RETRY_LIMIT + 1):
        try:
            raw_claim = raw_claims.invoke([
                SystemMessage(content=RESEARCH_SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=f"Query: {query}\n\nSearch results:\n{format_search_results(results)}"),
            ])
            return Claim(
                proposition=raw_claim.proposition,
                evidence=[(e.quote, e.source) for e in raw_claim.evidence],
                reasoning=raw_claim.reasoning,
            )
        except Exception:
            continue
    fallback_evidence = [("", r.get("url")) for r in results.get("results", [])]
    return Claim(proposition=FALLBACK_TEXT, evidence=fallback_evidence, reasoning="")

def constitutional_builder(query, docs):
    excerpts = "\n".join(f"{i + 1}. [{doc.metadata['reference']}] {doc.page_content}" for i, doc in enumerate(docs))
    raw_claim = raw_constitutional.invoke([
        SystemMessage(content=CONSTITUTIONAL_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}\n\nExcerpts:\n{excerpts}"),
    ])
    return Claim(
        proposition=raw_claim.proposition,
        evidence=[(doc.page_content, doc.metadata["reference"]) for doc in docs],
        reasoning=raw_claim.reasoning,
    )

def run_factual(query):
    results = search_tool.invoke(query)
    return claim_builder(query, results)

def run_perspective(pair):
    perspective_query, entity = pair
    query = f"{entity} {perspective_query}"
    results = search_tool.invoke(query)
    return claim_builder(query, results)

def run_constitutional(query):
    docs = constitution_index.similarity_search(query, k=2)
    return constitutional_builder(query, docs)

#final output for the research section. Maps queries to researched claims, to be synthesized
#by the core chat interface. Parallelism for independent queries to minimize latency
def research_output(inquiry):
    tasks = (
        [(run_factual, q) for q in inquiry.factual_inquiries]
        + [(run_perspective, p) for p in inquiry.perspective_inquiries]
        + [(run_constitutional, q) for q in inquiry.constitutional_inquiries]
    )
    if not tasks:
        return []

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(fn, arg) for fn, arg in tasks]
        claims = [f.result() for f in futures]

    return claims

def set_model(model_name): #allow user to pick btwn 3 openai models
    global llm, raw_claims, raw_constitutional
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    raw_claims = llm.with_structured_output(ClaimSchema)
    raw_constitutional = llm.with_structured_output(ConstitutionalSchema)
