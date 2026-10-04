"""Is `max_tokens` a hard cap on this route?

The compiler run produced a 2112-token response to a request with max_tokens=600
and finish_reason=stop. If the cap is not honoured, every token-based efficiency
number measured against it is under-stated, so this is checked directly.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from laec import complete  # noqa: E402

# Prompts chosen to invite long output, which is where a broken cap would show.
PROMPTS = [
    "Count slowly from 1 to 200, one number per line, with no commentary.",
    "Explain in detail, with many examples, how a hash table works.",
    "Write out the alphabet repeatedly until you are told to stop.",
    "List every two-letter combination of the English alphabet, one per line.",
    "Describe at length the history of computing from 1900 to 2020.",
]
CAP = 64


def main() -> int:
    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        res = complete({"model": "Qwen-Ambassador/Qwen3.8-Max"}, "", prompt, CAP)
        over = res["completion_tokens"] > CAP
        rows.append({
            "prompt": prompt[:50], "max_tokens": CAP,
            "completion_tokens": res["completion_tokens"],
            "finish_reason": res["finish_reason"],
            "exceeded_cap": over,
            "ratio": round(res["completion_tokens"] / CAP, 2),
            "latency_s": res["latency_s"],
        })
        print(f"{i}: cap={CAP} actual={res['completion_tokens']:>5} "
              f"ratio={rows[-1]['ratio']:>6} finish={res['finish_reason']} "
              f"{'EXCEEDED' if over else 'within cap'}")

    exceeded = sum(1 for r in rows if r["exceeded_cap"])
    summary = {
        "cap": CAP,
        "requests": len(rows),
        "exceeded": exceeded,
        "max_observed": max(r["completion_tokens"] for r in rows),
        "max_ratio": max(r["ratio"] for r in rows),
        "cap_is_hard": exceeded == 0,
    }
    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "max_tokens_probe.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
