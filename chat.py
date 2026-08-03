#Setup and imports
import re
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, OPENAI_MODEL
from prompts import CONTEXT_SUMMARY_SYSTEM_PROMPT, RELEVANCE_CHECK_SYSTEM_PROMPT, RESEARCH_INQUIRY_SYSTEM_PROMPT, CORE_CHAT_PROMPT
from research import research_output, FALLBACK_TEXT
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
)

#Context management:
#We initially just record all user and AI chat interactions in a list of strings.
#General rule of thumb; we don't want the context to exceed 10000 characters to avoid latency
#If we hit the character limit, we call the LLM to summarize context for a 50% character reduction
#We don't summarize every time, because that would cause latency. Instead, we do a big task, but after a LONG conversation
#We can only summarize so far. Every 5 summaries (50,000 characters), we just drop the oldest 20% of the context
context = []
summary_count = 0
CONTEXT_CHAR_LIMIT = 10000
NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*")

def trim_context(context):
    global summary_count
    if sum(len(c) for c in context) < CONTEXT_CHAR_LIMIT:
        return context
    elif summary_count < 5:
        numbered_entries = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(context))
        response = llm.invoke([
            SystemMessage(content=CONTEXT_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=numbered_entries),
        ])
        shortened = [
            NUMBERED_LINE_RE.sub("", line).strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
        if len(shortened) == len(context):
            context = shortened
        summary_count += 1
        return context
    else:
        cutoff = int(len(context) * 0.2)
        summary_count = 0
        return context[cutoff:]


#Chat interface:
#Chat is supplied the relevance checker first which rejects off-topic or inappropriate messages
#Rejection messages aren't sent through the research pipeline, saving on latency
#When messages are processed, they are sent to the research planner, which decides what needs to be researched before responding
#3 categories of research: factual claim, perspective pairs, and constitutional claims.
#When no perspective is specified we generalize to all perspectives to ensure a balanced view. Constitutional claims are sent to a RAG
accepting_chats = True
VIOLATION_LIMIT = 10
violation_count = 0

class RelevanceScore(BaseModel): #output schema that enforces strucutre for relevance judge
    politics_relevance: int = Field(ge=1, le=5, description="How pertinent the message is to politics, 1-5.")
    safe_language: int = Field(ge=1, le=5, description="How free of vulgar/inappropriate language the message is, 1-5.")

relevance_judge = llm.with_structured_output(RelevanceScore)

def check_relevance(txt, recent_context):
    try:
        if recent_context:
            context_block = "Recent turns:\n" + "\n".join(recent_context) + "\n\n"
        else:
            context_block = ""
        result = relevance_judge.invoke([
            SystemMessage(content=RELEVANCE_CHECK_SYSTEM_PROMPT),
            HumanMessage(content=f"{context_block}New message: {txt}"),
        ])
        return (result.politics_relevance, result.safe_language)
    except Exception:
        return None

def guardrail(txt):
    global violation_count, accepting_chats
    if not accepting_chats:
        return "This chat is either too inappropriate or off-topic, and has ended."
    scores = check_relevance(txt, context[-4:])
    if scores is None: #means some API didn't respond, so later chats aren't going to work
        return "API Error: please try again."
    if scores[0] < 3 or scores[1] < 3:
        violation_count += 1
        if violation_count >= VIOLATION_LIMIT: #we accept 10 off topic or inappropriate messages (chance at redemption) then shut down
            accepting_chats = False
            return "This chat is either too inappropriate or off-topic, and has ended."
        if scores[0] < 3 and scores[1] < 3:
            return "This chat is meant to be political and respectful. Please try again."
        elif scores[0] < 3:
            return "This chat is meant to focus on political discussion. I can chat about politics, but I can't help with productivity tasks or off-topic discussions. Please try again."
        else:
            return "This chat is meant to be respectful and appropriate. Please try again."
    else:
        return None

def collect_chat(txt): #if we dont screen a chat out we respond to it
    prescreened = guardrail(txt)
    if prescreened is not None:
        return prescreened
    return chat_respond(txt)


class PerspectivePair(BaseModel): #schema for perspective inquiries, which have both an issue and a person/entity
    inquiry: str = Field(description="The perspective query.")
    entity: str = Field(description="The person/group whose perspective is sought.")

class ResearchInquirySchema(BaseModel):
    factual_inquiries: list[str] = Field(default_factory=list, description="Concise, search-engine-ready factual queries.")
    perspective_inquiries: list[PerspectivePair] = Field(default_factory=list, description="Perspective queries, each paired with the person/group whose perspective is sought.")
    constitutional_inquiries: list[str] = Field(default_factory=list, description="Concise queries about constitutional text or meaning.")

class ResearchInquiry(BaseModel): #similar to Claims, we want perspective pairs as tuples, but OpenAI's structured output rejects tuple fields. So, we use the schema in the
    factual_inquiries: list[str] = Field(default_factory=list) #builder class and convert to string tuples in the actual version
    perspective_inquiries: list[tuple[str, str]] = Field(default_factory=list)
    constitutional_inquiries: list[str] = Field(default_factory=list)

    def is_empty(self):
        return not (self.factual_inquiries or self.perspective_inquiries or self.constitutional_inquiries)

research_planner = llm.with_structured_output(ResearchInquirySchema)

class SourceEntry(BaseModel):
    citation: str = Field(description="The citation marker used in the answer, e.g. '1'.")
    url: str = Field(description="The source URL for this citation.")

class ChatResponse(BaseModel):
    reasoning: str = Field(description="Internal step-by-step reasoning, not shown to the user.")
    answer: str = Field(description="The final user-facing response.")
    sources: list[SourceEntry] = Field(default_factory=list, description="Citation markers used in the answer, paired with their source URLs.")

chat_responder = llm.with_structured_output(ChatResponse)

def format_claims(claims): #formats claim objects to be sent to the model for final processing. Model can read strings not claim objects
    lines = []
    for c in claims:
        evidence_str = "; ".join(f'"{quote}" ({source})' for quote, source in c.evidence)
        lines.append(f"- Proposition: {c.proposition}\n  Evidence: {evidence_str}\n  Reasoning: {c.reasoning}")
    return "\n".join(lines)

def format_sources(sources): #same purpose as above
    grouped = {}
    order = []
    for s in sources:
        if s.citation not in grouped:
            grouped[s.citation] = []
            order.append(s.citation)
        grouped[s.citation].append(s.url)
    lines = [f"({c}) " + "; ".join(grouped[c]) for c in order]
    return "\n".join(lines)

last_reasoning = None

def chat_respond(txt): #sources info, structures into response
    global context, last_reasoning
    context = trim_context(context)
    context_block = "\n".join(context)

    raw_inquiry = research_planner.invoke([
        SystemMessage(content=RESEARCH_INQUIRY_SYSTEM_PROMPT),
        HumanMessage(content=f"Recent conversation:\n{context_block}\n\nNew message: {txt}"),
    ])
    inquiry = ResearchInquiry(
        factual_inquiries=raw_inquiry.factual_inquiries,
        perspective_inquiries=[(p.inquiry, p.entity) for p in raw_inquiry.perspective_inquiries],
        constitutional_inquiries=raw_inquiry.constitutional_inquiries,
    )

    claims = [] if inquiry.is_empty() else research_output(inquiry)
    failed_urls = [url for c in claims if c.proposition == FALLBACK_TEXT for _, url in c.evidence]
    claims = [c for c in claims if c.proposition != FALLBACK_TEXT]
    claims_block = format_claims(claims)

    response = chat_responder.invoke([
        SystemMessage(content=CORE_CHAT_PROMPT),
        HumanMessage(content=f"Conversation so far:\n{context_block}\n\nResearched claims:\n{claims_block}\n\nUser message: {txt}"),
    ])

    last_reasoning = (
        "The research pipeline triggered OpenAI's copyright filters. To read more, refer to the sources section"
        if failed_urls else response.reasoning
    )
    answer = response.answer
    sources_block = format_sources(response.sources)
    if failed_urls: #sources that failed quote-extraction still get listed, just without a quoted citation tying them to the answer text
        next_marker = len({s.citation for s in response.sources}) + 1
        extra_lines = "\n".join(f"({next_marker + i}) {url}" for i, url in enumerate(failed_urls))
        sources_block = f"{sources_block}\n{extra_lines}" if sources_block else extra_lines
    if sources_block:
        answer = f"{answer}\n\nSources:\n{sources_block}"

    context.append(f"User: {txt}")
    context.append(f"AI: {answer}")

    return answer

def set_model(model_name):
    global llm, relevance_judge, research_planner, chat_responder
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    relevance_judge = llm.with_structured_output(RelevanceScore)
    research_planner = llm.with_structured_output(ResearchInquirySchema)
    chat_responder = llm.with_structured_output(ChatResponse)


