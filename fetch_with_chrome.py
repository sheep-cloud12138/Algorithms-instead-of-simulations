from __future__ import annotations

import json
import sys
import time
import urllib.request

from websockets.sync.client import connect


def command(ws, command_id: int, method: str, params: dict | None = None) -> dict:
    ws.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
    while True:
        message = json.loads(ws.recv())
        if message.get("id") == command_id:
            return message


def evaluate(ws, command_id: int, expression: str) -> str:
    response = command(
        ws,
        command_id,
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    result = response.get("result", {}).get("result", {})
    return result.get("value", "")


def main() -> None:
    url = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else "9223"
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10) as response:
        targets = json.load(response)
    page = next(target for target in targets if target["type"] == "page")

    with connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        command(ws, 1, "Page.enable")
        command(ws, 2, "Runtime.enable")
        command(ws, 3, "Page.navigate", {"url": url})

        for _ in range(60):
            time.sleep(2)
            title = evaluate(ws, 10, "document.title")
            text = evaluate(ws, 11, "document.body ? document.body.innerText : ''")
            if title and "moment" not in title.lower() and "sciencedirect" not in title.lower():
                print("TITLE:", title)
                print("URL:", evaluate(ws, 12, "location.href"))
                print("TEXT_LENGTH:", len(text))
                print(text[:8000])
                return
        print("TIMEOUT title=", title, "text_length=", len(text))
        print(text[:2000])


if __name__ == "__main__":
    main()
