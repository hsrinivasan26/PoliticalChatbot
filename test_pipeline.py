"""Full-visibility test run of the research + response pipeline for a single query.

Run directly: python3 test_pipeline.py
"""
from langchain_core.messages import SystemMessage, HumanMessage

from chat import research_planner, ResearchInquiry, chat_responder, _format_claims, _format_sources
from prompts import RESEARCH_INQUIRY_SYSTEM_PROMPT, CORE_CHAT_PROMPT
from research import research_output

QUERY = "What happened in the 2023 debt ceiling negotiations, and how did Democrats and Republicans each frame the deal?"


def _header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def run_test(query):
    _header("1. INPUT QUERY")
    print(query)

    raw_inquiry = research_planner.invoke([
        SystemMessage(content=RESEARCH_INQUIRY_SYSTEM_PROMPT),
        HumanMessage(content=f"Recent conversation:\n\nNew message: {query}"),
    ])
    inquiry = ResearchInquiry(
        factual_inquiries=raw_inquiry.factual_inquiries,
        perspective_inquiries=[(p.inquiry, p.entity) for p in raw_inquiry.perspective_inquiries],
        constitutional_inquiries=raw_inquiry.constitutional_inquiries,
    )

    _header("2. RESEARCH INQUIRY BREAKDOWN")
    print("Factual inquiries:")
    for f in inquiry.factual_inquiries:
        print(f"  - {f}")
    print("Perspective inquiries:")
    for inq, entity in inquiry.perspective_inquiries:
        print(f"  - ({entity}) {inq}")
    print("Constitutional inquiries:")
    for c in inquiry.constitutional_inquiries:
        print(f"  - {c}")

    claims = [] if inquiry.is_empty() else research_output(inquiry)

    _header("3. CLAIM OBJECTS")
    for i, c in enumerate(claims, 1):
        print(f"--- Claim {i} ---")
        print(f"Proposition: {c.proposition}")
        print("Evidence:")
        for quote, source in c.evidence:
            print(f'  - "{quote}"\n    ({source})')
        print(f"Reasoning: {c.reasoning}")
        print()

    claims_block = _format_claims(claims)
    response = chat_responder.invoke([
        SystemMessage(content=CORE_CHAT_PROMPT),
        HumanMessage(content=f"Conversation so far:\n\nResearched claims:\n{claims_block}\n\nUser message: {query}"),
    ])

    answer = response.answer
    sources_block = _format_sources(response.sources)
    if sources_block:
        answer = f"{answer}\n\nSources:\n{sources_block}"

    _header("4. FINAL TEXT OUTPUT")
    print(answer)
    print()


if __name__ == "__main__":
    run_test(QUERY)
