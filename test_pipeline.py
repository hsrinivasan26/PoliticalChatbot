"""Interactive CLI chat with the political events chatbot.

Run directly: python3 test_pipeline.py
Type 'exit' or 'quit' to stop.
"""
from chat import collect_chat

if __name__ == "__main__":
    print("Political events chatbot. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue

        response = collect_chat(user_input)
        print(f"\nAssistant: {response}\n")
