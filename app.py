import gradio as gr

import chat
import research
from config import OPENAI_MODEL

SOURCES_MARKER = "\n\nSources:\n"


def split_sources(answer):
    if SOURCES_MARKER in answer:
        main, sources = answer.split(SOURCES_MARKER, 1)
        return main, sources
    return answer, None


def build_message(main_answer, sources_text, reasoning_text):
    parts = [main_answer]
    if reasoning_text:
        parts.append(f"<details>\n<summary>Show reasoning</summary>\n\n{reasoning_text}\n\n</details>")
    if sources_text:
        parts.append(f"<details>\n<summary>Show sources</summary>\n\n{sources_text}\n\n</details>")
    return "\n\n".join(parts)


def respond(message, history, selected_model, context_limit):
    if selected_model:
        chat.set_model(selected_model)
        research.set_model(selected_model)
    chat.CONTEXT_CHAR_LIMIT = int(context_limit)

    yield "🤔 Thinking..."

    prescreened = chat.guardrail(message)
    if prescreened is not None:
        yield prescreened
        return

    answer = chat.chat_respond(message)
    main_answer, sources_text = split_sources(answer)
    reasoning_text = chat.last_reasoning

    yield build_message(main_answer, sources_text, reasoning_text)


model_dropdown = gr.Dropdown(
    choices=["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"],
    value=OPENAI_MODEL,
    allow_custom_value=True,
    label="Model",
)

context_slider = gr.Slider(
    minimum=2000,
    maximum=20000,
    value=chat.CONTEXT_CHAR_LIMIT,
    step=1000,
    label="Context/memory amount (characters kept before summarizing) - less is faster, more is higher quality",
)

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[model_dropdown, context_slider],
    title="Political Events Chatbot",
)

if __name__ == "__main__":
    demo.launch()
