from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, TAVILY_API_KEY, OPENAI_MODEL
from prompts import RESEARCH_SYNTHESIS_SYSTEM_PROMPT

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
)

search_tool = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=10,
)


class Claim(BaseModel):
    proposition: str
    evidence: list[tuple[str, str]]
    reasoning: str


class _EvidencePair(BaseModel):
    quote: str = Field(description="A direct quotation from the source material.")
    source: str = Field(description="The link to the source, or an amendment/article number if constitutional.")

class _ClaimSchema(BaseModel):
    proposition: str = Field(description="The claim being made, answering the query.")
    evidence: list[_EvidencePair] = Field(default_factory=list, description="Quotes paired with their sources supporting the proposition.")
    reasoning: str = Field(description="How the evidence supports the proposition.")

claim_synthesizer = llm.with_structured_output(_ClaimSchema)


def _format_search_results(results):
    lines = []
    for i, r in enumerate(results.get("results", [])):
        lines.append(f"{i + 1}. Title: {r.get('title')}\n   URL: {r.get('url')}\n   Content: {r.get('content')}")
    return "\n".join(lines)

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

def research_output(inquiry):
    claims = []

    for factual_query in inquiry.factual_inquiries:
        results = search_tool.invoke(factual_query)
        claims.append(_synthesize_claim(factual_query, results))

    for perspective_query, entity in inquiry.perspective_inquiries:
        query = f"{entity} {perspective_query}"
        results = search_tool.invoke(query)
        claims.append(_synthesize_claim(query, results))

    # constitutional_inquiries not yet handled - requires a constitutional RAG pipeline.

    return claims
