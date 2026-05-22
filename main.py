"""CLI: run research pipeline then optional follow-up Q&A over the same index."""

from src.agent import answer_follow_up, run_agent


def main() -> None:
    topic = input("Enter topic: ").strip()
    result = run_agent(topic)
    print("\n" + result)

    while True:
        q = input("\nAsk (or exit): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        print("\n" + answer_follow_up(q))


if __name__ == "__main__":
    main()
