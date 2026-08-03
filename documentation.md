# PoliticalChatbot: Documentation


## config.py

Safety layer to prevent API errors and such. Calls _require on all the API keys so if OpenAI or Tavily or LangSmith has some rate limit, key error, etc., we have visibility into that as opposed to a downstream error that might be difficult to diagnose.

## app.py

Minimal frontend via Gradio. The response yielded is the one output but chat.guardrail, which in turn calls collect_chat (which in turn calls chat_respond). A lot of the frontend/UX choices were inspired by the Claude.ai chatbot. Specifically, the ability to toggle between models with a dropdown, and the ability to reveal chain-of-thought internal reasoning. I added two more features on the frontend that are tailored to this specific use-case: first, context is an important layer, because it represents the tradeoff between latency (preserving more context) and richness of response. We outsource this decision to the user, because only they know the intended complexity of their chat. The other thing they're able to do is view sources, which addresses the question around hallucination and transparency. The entire chat follows the rule-of-thumb that every claim made must be sourced, so users can sanity-check the claims themselves by reading through the sources provided in bibliography format.

## chat.py

**Context management** is a running list of "User: ..." / "AI: ..." strings, summarized by the LLM once the transcript crosses a character budget (`CONTEXT_CHAR_LIMIT`), which is 10000 characters by default (following the rule-of-thumb that >2000 word system prompts aren't very good). If we summarize every turn, that takes way too long, so we summarize only when we hit the character threshold. After five summarization passes without the conversation ending, trim_context() gives up trying to compress and instead just drops the oldest 20% of the transcript, because if we compress beyond that we would start losing meaning.

**The guardrail** consists of check_relevance() which sends the new message (along with the last few turns of context, so a short reply like "Oh, ok." can be judged as a continuation) to an LLM judge that returns an output schema of two 1–5 scores: political relevance and language safety. The schema is enforced by OpenAI's structured-output so we can query it programmatically-i.e. it's not a natural-language string. Structured outputs are arguably the most important prompt engineering tool used in the whole project.

guardrail() then makes a policy decision on top of those scores: rather than ending the conversation on a single off-topic message or resetting the count on every good message, it tracks a **lifetime violation count** that only ever goes up, and permanently locks the chat (accepting_chats = False) once it hits ten (this can be configured to be higher or lower). After this point, we also no longer send any requests to the LLM so user's can't continue to drain API credits.

**Research planning** is where the bot decides what it needs to go find out before it's allowed to answer. chat_respond( sends the message to research_planner, which returns a `ResearchInquiry` broken into three categories: factual_inquiries, perspective_inquiries (which are PAIRS, including whose perspective is sought), and constitutional_inquiries (routed to the RAG instead of the web). This output schema helps with a few things. Firstly the bias mitigation: when the user specifies a certain perspective, the research planner is instructed to return that perspective. But when they don't, it is instructed to duplicate the query across every perspective in the political spectrum. This is a really important safeguard, bc it means we're not relying on the search engine algorithm to produce balanced results--we decide up-front what ALL the relevant perspectives are, and look for them individually rather than just hoping they will show up in Tavily's sources.

There's another mechanism at work here, which is that ResearchInquiry and ResearchInquirySchema are different classes. This could seem odd but it has to do with a property of OpenAI's structured outputs. Structured outputs cannot return fields that consist of tuples (like the perspective pairs we want), but they CAN return OTHER structured outputs. So, we initially get the perspective pairs in the form of a PerspectivePair object (as a subfield of ResearchInquirySchema) and then convert it to a tuple in the actual ResearchInquiry to send to Tavily.

**Response synthesis** relies on chain-of-thought: ChatResponse has reasoning, answer, and sources as three separate structured fields, and the model is forced to work through five explicit reasoning steps (what's being asked, whether the supplied claims cover it, whether the neutrality rule is actually satisfied, how contradictions are being resolved, what's still uncertain) before it's allowed to produce the answer field. The prompt tells the model explicitly that it's "an impartial curator of research that has already been conducted, not a source of new claims," so anything substantive in the answer has to trace back to a Claim object that was actually retrieved, NOT the model's own background knowledge.

## research.py 

Every inquiry maps to a Claim: a proposition, a list of (quote, source) evidence pairs, and reasoning connecting them. That claim-evidence-reasoning shape (I got this from my time as a debater) is the enforced structure that keeps the bot from ever asserting something without a traceable source. Because factual, perspective, and constitutional inquiries are all independent of one another (they never access the same info), research_output() can send them to separate threads in a ThreadPoolExecutor which allows us to run research threads in parallel and minimize latency. In my testing, this resulted in a ~3x improvement in response speed.

Web-sourced claims go through claim_builder(), which hands raw Tavily results to an LLM and asks it to extract a single central proposition, supporting quotes, and reasoning with a source credibility ranking (.gov/.edu, then print news, then broadcast/online news, then opinion, with forums, social media, and Wikipedia excluded outright) built in. There is still an occasional error here, which has to do with OpenAI's filters against producing copyrighted content (prompt is instructed to quote verbatim, which OpenAI doesn't like sometimes). Quoting verbatim is still a worthwhile tradeoff and in these situations we just fall back on the model's response and link to a bunch of relevant sources, so UX is still pretty good.

## rag.py

Constitutional questions get treated differently from general political questions on purpose: the Constitution is a fixed, authoritative, unchanging text, so instead of sending it to a general web search and hoping for a reliable source, it's indexed once into a local FAISS vector store and retrieved directly. The PDF gets parsed, stripped of its repeated running header, de-hyphenated across line wraps, and split by Article/Amendment headers into named sections.

The first version chunked the Constitution one Article or Amendment at a time, but Article I is enormous and covers a dozen unrelated powers of Congress in one block, so a narrow query like "what is the Commerce Clause" had to compete against the embedding of the entire rest of Article I, and consistently lost the meaning of the whole query within a mess of different clauses. The fix consisted of the function split_into_subsections() which further splits each Article or Amendment body on its literal "SECTION N" markers, so a query about the Commerce Clause now gets matched to the subsection covering interstate commerce. This makes RAG retrieval much more reliable.

## prompts.py

Every LLM call in the system is governed by a prompt in this one file, and a few philosophies run consistently through all of them. First, RELEVANCE_CHECK_SYSTEM_PROMPT scores political relevance and language safety as a reasoning task, enforcing a structured output on the relevance judge. This eliminates the possibility of a LLM returning "TRUE, because..." and erroring on a boolean calculation or something like that. It also clearly delineates the difference between political and inappropriate queries--someone using insulting language towards their political enemies is relevant, but not safe.

Second, neutrality is enforced structurally by CORE_CHAT_PROMPT by idenfiying relevant perspectives BEFORE the research is conducted rather than just trying to select neutral perspectives out of research that is already there. CORE_CHAT_PROMPT also enforces our source quality hierarchy and gives the explicit instruction to the model not to draw on its own internal knowledge to answer queries, but only the research pipeline. The user can sanity-check this by noticing that every assertion is cited with a source that comes from the research pipeline.

## test_pipeline.py

Each of the eighteen test cases is a full multi-turn conversation run through the real guardrail() → chat_respond() pipeline against live APIs, printing the full transcript including the normally-hidden reasoning field for every turn that actually reached synthesis. The cases were chosen to cover every one of the five scenarios named explicitly in the project spec, plus deliberate probes for important KPIs: whether a loaded, biased question gets mirrored back or its bias is corrected, whether an explicit "what does X think" question stays scoped to X while an unscoped one spreads across the spectrum, whether debunked claims get pushed back on rather than validated (no sycophancy), and whether the ten-violation lockout actually holds once tripped. On top of this local suite, four LangSmith online evaluators (Hallucination, Bias & Fairness, Source Reliability, and OpenAI Error Rate) run continuously against live production traces, giving a second, independent signal on the same qualities the local test cases probe for turn-by-turn.

## STILL WORKING ON:
- Minimizing the OpenAI copyright errors--engineering a fallback wherein copyrighted text triggers a paraphrasing function rather than just returning to model defaults
- More evaluation metrics--specifically, RAG retrieval accuracy. Want visibility into whether the embeddings model works well, but it doesn't appear in the LangSmith "run" object so it needs to be built internally

---

## Rubric Coverage

| Rubric Criterion | Feature | How It's Addressed |
|---|---|---|
| Multi-step reasoning chains for political info validation | ResearchInquiry + ChatResponse.sources | The research pipeline identifies all relevant perspectives to seek and sends them to distinct research queries. Queries are answered with at least one-to-one claim/evidence ratio, making EVERY claime citable. |
| Decision-making frameworks for handling uncertainty | Chain-of-Thought Prompting | Uncertainty is surfaced in the 5-step CoT process, no answer given without sources, links given for further reading |
| Perspective-taking and balanced analysis | perspective_inquiries (ResearchInquiry) | When no specific perspective is named, the research planner must identify every relevant perspective across the political spectrum and research each one separately. |
| Conversational state management | context / trim_context() | Maintains a running transcript that gets sent to the system prompt so that it has memory of past interactions. Trims irrelevant context where needed to minimize latency. |
| Intelligent boundary detection and scope management | guardrail() / check_relevance() | Scores every message for political relevance and safety via a structured-output LLM judge (context included, so "Oh, ok"-type messages acceptable) with no keyword or regex list involved. |
| Use of information-access tools for factual grounding | run_factual() (Tavily) | Routes every factual inquiry to a live web search rather than the model's own memory. One-to-one map btwn claims and sources |
| Neutrality enforced through prompt reasoning | CORE_CHAT_PROMPT trait ordering, PerspectivePair schema, evals | Ranks impartiality as a top-priority trait the model must actively satisfy, while separating diverse PerspectiveInquiries independently to make sure all are represented. LLM-as-judge evals on this post-run|
| Multi-perspective analysis framework | PerspectivePair schema | Structures perspective research as (inquiry, entity) pairs so each side of an issue is researched as its own independent query. |
| Bias detection and self-correction (meta-cognitive reflection) | Reasoning step 3 + Langsmith evals | Reasoning requires the model to expose its thought process at every step of the response generation. It is forced to demonstrate its neutrality back to the user. Evals also investigate bias post-run |
| Balanced information synthesis | RESEARCH_SYNTHESIS_SYSTEM_PROMPT source tiering | Ranks sources by type (.gov/.edu > print news > broadcast/online > opinion) rather than by political leaning, with loopholes built in if a user wants to know about a certain specific perspective |
| Systematic fairness on partisan issues | Claim schema + PerspectivePair schema | Forces every proposition to carry its own evidence and reasoning, so no assertion can enter a response without a traceable source. When we're not asked for a specific entity's opinion, ResearchInquiry is forced to consider them all.|
| Reasoning-based, non-keyword request classification | check_relevance() | Uses a structured-output LLM judge instead of any hardcoded keyword or pattern list — directly avoids the project's stated immediate-failure condition. |
| Graceful boundary management with explained reasoning | guardrail() tiered messages | Returns a specific explanation depending on whether the message failed on relevance, safety, or both, instead of one generic refusal. Users get to "redeem themselves" |
| Conversational repair when discussion goes off-track | Guardrail redirect messages | Every rejection invites the user to continue ("Please try again"), keeping the conversation open rather than shutting it down. Also gives reason for message reject |
| Reasoning through ambiguous or borderline topics | RELEVANCE_CHECK_SYSTEM_PROMPT context-awareness | Scores new messages as a continuation of recent turns, so short or ambiguous replies are judged in conversational context, not isolation. |
| Principled, proportionate boundary enforcement | Lifetime violation counter | Tolerates repeated off-topic drift up to ten violations before permanently ending a chat, rather than terminating on the first offense. |
| Evaluation of reasoning quality and consistency | test_pipeline.py + LangSmith evals | Surfaces the model's actual reasoning for every turn in a runnable transcript, plus continuous LLM-judge scoring of hallucination, bias, source quality, and API consistency on live traces. |
| Bias/neutrality testing across conversation contexts | Bias & Fairness LangSmith evaluator + dedicated test cases | Includes conversations built specifically to bait the bot into mirroring a loaded political framing, and scores production traces for the same. |
| Benchmarks for conversational intelligence and boundary management | Guardrail-focused test cases | Covers subtle off-topic drift, subtle inappropriate framing, and the 10-violation lockout, not just obvious rejections. |
| Avoiding keyword-based / rule-based classification (immediate-failure criteria) | check_relevance() / guardrail() | Every scope and safety decision is made by an LLM reasoning over the message and its context — no keyword list exists anywhere in the classification path. |

---

## Installation & Setup

**Prerequisites**
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://tavily.com/) (free tier is enough)
- (Optional) A [LangSmith API key](https://smith.langchain.com/) if you want tracing/evaluation

**1. Clone the repo and install dependencies**

```bash
git clone <this-repo-url>
cd PoliticalChatbot
pip install gradio langchain langchain-openai langchain-community langchain-tavily langsmith faiss-cpu pypdf python-dotenv pydantic
```

**2. Create a `.env` file** in the project root with:

```
OPENAI_API_KEY=your-openai-key-here
TAVILY_API_KEY=your-tavily-key-here

# Optional - only needed if you want LangSmith tracing/evals
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-key-here
LANGSMITH_PROJECT=PoliticalChatbot
```

`OPENAI_MODEL` can also be set here to override the default model, but this is optional — it's also changeable live from the frontend's model dropdown.

**3. Make sure `constitution.pdf` is present** in the project root (it's included in this repo). The first time `rag.py` runs, it parses the PDF and builds a local FAISS index under `constitution_index/` — this only happens once; every run after that loads the saved index instantly.

**4. Run the app**

```bash
python app.py
```

This starts a local Gradio server (prints a `http://127.0.0.1:7860`-style URL to the terminal) — open it in a browser to chat.

**5. Run the test suite** (optional, but useful to sanity-check a fresh setup)

```bash
python test_pipeline.py        # interactive menu of all 18 test cases
python test_pipeline.py 4      # run a specific case by number, full transcript printed
```
