import requests
import json
import time
from typing import List, Union, Generator, Dict, Any, Optional

class MLCClient:
    def __init__(
        self,
        base_url: str = "http://192.168.1.107:8000",
        model_name: str = "/mlc/models/llm/Llama-3.2-1B-q4f16_1-MLC",
        default_timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_timeout = default_timeout

    def _normalize_messages(self, messages: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Accept either a raw user string or a list of messages in OpenAI format.
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        # assume user passed already-correct messages list
        return messages

    def generate(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Non-streaming call (synchronous). Returns full string content.
        """
        timeout = timeout or self.default_timeout
        msgs = self._normalize_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        j = r.json()
        # Safe navigation of structure
        choice = j.get("choices", [{}])[0]
        # either `message` or `delta` depending on API; for non-stream expect `message`
        message = choice.get("message") or {}
        return message.get("content", "")

    def generate_stream(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: Optional[int] = None,
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Streaming generator. Yields events (dict) for each SSE message.

        Each yielded event example:
          {"type": "delta", "delta": "partial text"}
          {"type": "role", "role": "assistant"}  # sometimes role appears first
          {"type": "done", "content": "<final full text>", "usage": {...} }

        Finally returns a dict with final content & usage (see generator's return value if you want it).
        """

        timeout = timeout or self.default_timeout
        msgs = self._normalize_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": stream_options or {"include_usage": True},
        }

        url = f"{self.base_url}/v1/chat/completions"
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            try:
                r.raise_for_status()
            except Exception as e:
                # yield an error event and stop
                yield {"type": "error", "error": str(e), "status_code": getattr(r, "status_code", None)}
                return {"content": "", "usage": None}

            # Buffer lines for an event (SSE events end with a blank line)
            event_lines: List[str] = []
            full_text_parts: List[str] = []
            usage_info: Optional[Dict[str, Any]] = None

            # iterate over raw lines from server-sent-events
            for raw_line in r.iter_lines(decode_unicode=True):
                # iter_lines yields '' on event boundary (blank line)
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if line == "":
                    # end of an SSE event -> process accumulated lines
                    if not event_lines:
                        continue
                    event_data = "\n".join(event_lines)
                    event_lines = []

                    # Sometimes the data is exactly "[DONE]"
                    if event_data.strip() == "[DONE]":
                        # final event marker
                        final_content = "".join(full_text_parts)
                        yield {"type": "done", "content": final_content, "usage": usage_info}
                        # Return final result as generator return value
                        return {"content": final_content, "usage": usage_info}

                    # Usually event_data contains JSON (possibly multiple JSON lines concatenated)
                    # Try to parse each JSON chunk if multiple lines
                    try:
                        # If multiple JSON objects concatenated with newlines, parse each
                        for sub in event_data.splitlines():
                            sub = sub.strip()
                            if not sub:
                                continue
                            # server may send prefix like "data: {...}" - ensure pure JSON
                            if sub.startswith("data:"):
                                sub = sub[len("data:"):].strip()
                            # parse JSON
                            data_json = json.loads(sub)
                            # extract useful parts
                            choices = data_json.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            # streaming delta content (per OpenAI style)
                            delta = choice.get("delta", {})
                            if delta:
                                # delta can contain role or content
                                if "role" in delta:
                                    yield {"type": "role", "role": delta["role"]}
                                if "content" in delta:
                                    chunk = delta["content"]
                                    full_text_parts.append(chunk)
                                    yield {"type": "delta", "delta": chunk}
                            # sometimes final chunk provides a `message` instead of delta
                            if "message" in choice:
                                msg = choice["message"]
                                content = msg.get("content")
                                if content:
                                    full_text_parts.append(content)
                                    yield {"type": "delta", "delta": content}
                            # capture usage if present
                            if "usage" in data_json:
                                usage_info = data_json["usage"]
                    except json.JSONDecodeError:
                        # If we can't parse JSON, yield raw data for debugging
                        yield {"type": "raw", "data": event_data}
                    continue

                # Not an event boundary: accumulate data lines (often "data: <json>")
                # Some SSE implementations send "data: " prefix on every line.
                event_lines.append(line)

            # If we exit the loop without receiving [DONE], still return accumulated content
            final_content = "".join(full_text_parts)
            if final_content:
                yield {"type": "done", "content": final_content, "usage": usage_info}
                return {"content": final_content, "usage": usage_info}

            # If no content and no done marker, yield an error
            yield {"type": "error", "error": "stream_ended_without_done"}
            return {"content": "", "usage": None}


# ========== Example usage ==========

if __name__ == "__main__":
    mlcc = MLCClient()

    # Example messages (you can pass a string instead)
    messages = [
        {"role": "system", "content": "Bạn là 1 trợ lý thành thạo Tiếng Việt."},
        {"role": "user", "content": "Xin chào!,Hãy giới thiệu Việt Nam đi."},
    ]

    # Non-streaming
    try:
        text = mlcc.generate(messages, temperature=0.2, max_tokens=128)
        print("== Non-streaming result ==\n", text)
    except Exception as e:
        print("generate() failed:", e)

    # Streaming
    print("\n== Streaming result ==\n")
    accum = []
    try:
        for event in mlcc.generate_stream(messages, temperature=0.2, max_tokens=128):
            t = event.get("type")
            if t == "delta":
                # print incremental delta
                print(event["delta"], end="", flush=True)
                accum.append(event["delta"])
            elif t == "role":
                # role announcement
                print(f"\n[role={event['role']}] ", end="", flush=True)
            elif t == "raw":
                print("\n[raw event]", event["data"])
            elif t == "error":
                print("\n[error]", event.get("error"))
            elif t == "done":
                print("\n\n[done] usage:", event.get("usage"))
                break
        final = "".join(accum)
        print("\n\nFinal assembled text:\n", final)
    except Exception as e:
        print("Streaming failed:", e)

