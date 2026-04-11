#!/usr/bin/env python3
"""
Tool-calling accuracy battery for Qwen3.5-35B-A3B-q4km on llama-server.

Runs a fixed set of tool-call prompts and grades each response against
expected behavior. Outputs a pass/fail summary and saves full responses
to tool_call_results.json for inspection.

Usage:
    llama-server --jinja --chat-template-file <Qwen3.5-4B.jinja> -m <model>
    python3 tool_call_battery.py [--port 9099]
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

TOOLS = {
    "weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country/state, e.g. 'San Francisco, CA'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic on two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "op": {
                        "type": "string",
                        "enum": ["add", "sub", "mul", "div"],
                        "description": "Operation to perform",
                    },
                },
                "required": ["a", "b", "op"],
            },
        },
    },
    "stock_price": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL'",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    "file_op": {
        "type": "function",
        "function": {
            "name": "file_op",
            "description": "Read, write, or append to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "append"],
                    },
                    "path": {"type": "string", "description": "File path"},
                    "content": {
                        "type": "string",
                        "description": "Content for write/append (ignored for read)",
                    },
                },
                "required": ["operation", "path"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return result snippets",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
}


def T(*names):
    return [TOOLS[n] for n in names]


# Each test: (id, prompt, tools_available, expected)
# expected keys:
#   call_expected: bool — should the model emit a tool_call at all
#   function: str — expected function name (when call_expected=True)
#   required_args: dict — args that MUST appear with the given values
#   forbidden_args: list — args that must NOT appear (optional)
#   content_contains: list[str] — if call_expected=False, strings expected in content
TESTS = [
    (
        "weather_simple",
        "What is the weather in San Francisco, CA?",
        T("weather"),
        {
            "call_expected": True,
            "function": "get_current_weather",
            "required_args": {"location": "San Francisco, CA"},
        },
    ),
    (
        "weather_unusual_city",
        "Tell me the current weather in Reykjavik, Iceland.",
        T("weather"),
        {
            "call_expected": True,
            "function": "get_current_weather",
            "required_args": {"location": "Reykjavik, Iceland"},
        },
    ),
    (
        "calc_multi_arg",
        "Compute 47 multiplied by 13 using the calculator.",
        T("calculator"),
        {
            "call_expected": True,
            "function": "calculator",
            "required_args": {"a": 47, "b": 13, "op": "mul"},
        },
    ),
    (
        "calc_division",
        "Please divide 144 by 12 using the calculator tool.",
        T("calculator"),
        {
            "call_expected": True,
            "function": "calculator",
            "required_args": {"a": 144, "b": 12, "op": "div"},
        },
    ),
    (
        "pick_weather_from_two",
        "What's the weather like in Paris, France right now?",
        T("weather", "stock_price"),
        {
            "call_expected": True,
            "function": "get_current_weather",
            "required_args": {"location": "Paris, France"},
        },
    ),
    (
        "pick_stock_from_two",
        "Look up the current price of Apple stock.",
        T("weather", "stock_price"),
        {
            "call_expected": True,
            "function": "get_stock_price",
            "required_args": {"ticker": "AAPL"},
        },
    ),
    (
        "file_enum_arg",
        "Please read the file at /etc/hostname.",
        T("file_op"),
        {
            "call_expected": True,
            "function": "file_op",
            "required_args": {"operation": "read", "path": "/etc/hostname"},
        },
    ),
    (
        "file_write_content",
        "Write the text 'hello world' to /tmp/note.txt.",
        T("file_op"),
        {
            "call_expected": True,
            "function": "file_op",
            "required_args": {
                "operation": "write",
                "path": "/tmp/note.txt",
                "content": "hello world",
            },
        },
    ),
    (
        "search_optional_arg",
        "Search the web for 'llama.cpp speculative decoding' and limit results to 3.",
        T("search"),
        {
            "call_expected": True,
            "function": "web_search",
            "required_args": {
                "query": "llama.cpp speculative decoding",
                "max_results": 3,
            },
        },
    ),
    (
        "no_tool_needed_math",
        "What is 2 plus 2? Please answer directly without using any tools.",
        T("calculator", "weather"),
        {
            "call_expected": False,
            "content_contains": ["4"],
        },
    ),
    (
        "no_tool_available",
        "What's the weather in Berlin?",
        T("calculator", "stock_price"),
        {
            # Model has no weather tool; it should either say it can't
            # call one or answer from knowledge. A tool call to the wrong
            # function would be a failure.
            "call_expected": False,
            "content_contains": [],  # just check no wrong tool_call
        },
    ),
    (
        "pick_correct_of_many",
        "Read the contents of the file /var/log/syslog please.",
        T("weather", "stock_price", "file_op", "search", "calculator"),
        {
            "call_expected": True,
            "function": "file_op",
            "required_args": {"operation": "read", "path": "/var/log/syslog"},
        },
    ),
]


def request(port, messages, tools, *, max_tokens=512, temperature=0.0):
    body = {
        "model": "qwen35",
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "seed": 42,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def arg_match(actual, expected):
    """Loose equality: numbers compare by value, strings case-insensitive substring."""
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(actual).strip().lower() == expected.strip().lower() or \
               expected.strip().lower() in str(actual).strip().lower()
    return actual == expected


def grade(test_id, prompt, expected, response):
    issues = []
    try:
        choice = response["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError):
        return False, ["malformed response"]

    tool_calls = msg.get("tool_calls") or []

    if expected["call_expected"]:
        if not tool_calls:
            return False, [f"expected tool call, got content: {msg.get('content','')[:80]!r}"]
        tc = tool_calls[0]["function"]
        if tc["name"] != expected["function"]:
            issues.append(f"wrong function: got {tc['name']!r}, want {expected['function']!r}")
        try:
            args = tc["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return False, ["arguments not valid JSON"]
        for k, v in expected["required_args"].items():
            if k not in args:
                issues.append(f"missing arg {k!r}")
                continue
            if not arg_match(args[k], v):
                issues.append(f"arg {k!r}: got {args[k]!r}, want {v!r}")
    else:
        if tool_calls:
            wrong = tool_calls[0]["function"]["name"]
            issues.append(f"unexpected tool call: {wrong!r}")
        else:
            content = msg.get("content", "") or ""
            for needle in expected.get("content_contains", []):
                if needle.lower() not in content.lower():
                    issues.append(f"content missing {needle!r}")
    return len(issues) == 0, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--out", default="/tmp/tool_call_results.json")
    args = ap.parse_args()

    results = []
    pass_count = 0

    print(f"Running {len(TESTS)} tool-call tests against 127.0.0.1:{args.port}\n")
    t0 = time.time()

    for i, (tid, prompt, tools, expected) in enumerate(TESTS, 1):
        print(f"[{i:2d}/{len(TESTS)}] {tid} ... ", end="", flush=True)
        t_start = time.time()
        try:
            resp = request(args.port, [{"role": "user", "content": prompt}], tools)
        except Exception as e:
            print(f"FAIL (request error: {e})")
            results.append({"id": tid, "passed": False, "issues": [str(e)], "elapsed": 0})
            continue
        dt = time.time() - t_start
        ok, issues = grade(tid, prompt, expected, resp)
        if ok:
            pass_count += 1
            print(f"PASS ({dt:.1f}s)")
        else:
            print(f"FAIL ({dt:.1f}s): {'; '.join(issues)}")
        results.append({
            "id": tid,
            "prompt": prompt,
            "passed": ok,
            "issues": issues,
            "elapsed": dt,
            "response": resp,
        })

    total_dt = time.time() - t0
    print()
    print(f"=== {pass_count}/{len(TESTS)} passed in {total_dt:.1f}s ===")
    with open(args.out, "w") as f:
        json.dump({"results": results, "pass_count": pass_count, "total": len(TESTS)}, f, indent=2)
    print(f"Full results: {args.out}")
    sys.exit(0 if pass_count == len(TESTS) else 1)


if __name__ == "__main__":
    main()
