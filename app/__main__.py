"""Run a prompt against the local llama.cpp server."""

import argparse
import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Explain a qubit in one sentence.")
    args = parser.parse_args()

    body = {
        "model": "nemotron-jetson",
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.2,
        "max_tokens": 96,
        "stream": False,
    }
    request = Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as error:
        parser.exit(1, f"Request failed: {error}. Check the server and its log.\n")

    answer = result["choices"][0]["message"].get("content")
    if not answer or not answer.strip():
        parser.exit(1, "Nemotron returned an empty answer. Check the server log.\n")
    print(answer.strip())


if __name__ == "__main__":
    main()
