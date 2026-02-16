from dotenv import load_dotenv

from anthropic_client import ask_claude


def main() -> None:
    load_dotenv()
    prompt = input("Your prompt: ").strip()
    if not prompt:
        print("Prompt is empty.")
        return
    try:
        answer = ask_claude(prompt)
        print("\nClaude:\n")
        print(answer)
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
