"""Qwen semantic relevance scorer and deterministic offline baseline."""
import json
import math
import os
import time


class QwenRelevanceScorer:
    def __init__(self, client=None, model=None, api_key=None, base_url=None,
                 timeout=45.0, max_retries=2):
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
        self.base_url = (
            base_url or os.getenv("QWEN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.request_count = 0
        self.last_latency_s = 0.0
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
            if not key:
                raise RuntimeError("Qwen API key missing. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
            self.client = OpenAI(
                api_key=key, base_url=self.base_url,
                timeout=float(timeout), max_retries=int(max_retries),
            )

    @staticmethod
    def parse_response(content, candidates):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen returned an empty response")
        text = content.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == fence:
                lines.pop()
            text = "\n".join(lines).strip()
        data = json.loads(text)
        expected = {item["id"] for item in candidates}
        if not isinstance(data, dict) or set(data.keys()) != expected:
            received = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise ValueError(
                "Qwen candidate IDs mismatch: expected={}, received={}".format(
                    sorted(expected), received
                )
            )
        result = {}
        for candidate_id, item in data.items():
            if not isinstance(item, dict) or "relevance" not in item:
                raise ValueError("Invalid relevance payload for {}".format(candidate_id))
            score = float(item["relevance"])
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError("Relevance for {} must be finite and in [0, 1]".format(candidate_id))
            result[candidate_id] = {
                "relevance": score,
                "rationale": str(item.get("rationale", ""))[:400],
            }
        return result

    def score(self, task, candidates, belief):
        prompt = {
            "task_instruction": task,
            "current_state": belief.summary(),
            "candidate_skills": candidates,
            "requirements": [
                "Score each candidate as the NEXT high-level skill toward the task.",
                "Use the current symbolic goal predicates and held-object state.",
                "Prefer skills that make progress on the earliest unsatisfied goal.",
                "Do not estimate physical feasibility; a separate Can module handles it.",
                "Do not invent object names or candidate IDs.",
                "Return every candidate ID exactly once.",
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Say component of a SayCan planner for the LIBERO "
                    "robot manipulation benchmark. Rank semantic relevance only. "
                    "Return a JSON object mapping every candidate ID to an object "
                    "with numeric relevance in [0,1] and a concise rationale."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        self.request_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=1000,
                messages=messages,
            )
        finally:
            self.last_latency_s = time.perf_counter() - started
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Qwen returned no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return self.parse_response(content, candidates)


class KeywordRelevanceScorer:
    """Offline baseline; never silently substituted for Qwen."""
    def __init__(self):
        self.request_count = 0
        self.last_latency_s = 0.0
        self.model = "keyword-baseline"
        self.base_url = None

    def score(self, task, candidates, belief):
        first = belief.first_unsatisfied_goal()
        expected = None
        expected_args = {}
        if first:
            predicate = first.get("predicate", {})
            name = predicate.get("name", "")
            args = predicate.get("args", [])
            if name in ("on", "in") and len(args) >= 2:
                expected = (
                    "Pick" if belief.held_object != args[0]
                    else ("PutInside" if name == "in" else "PutOn")
                )
                expected_args = (
                    {"object": args[0]} if expected == "Pick"
                    else {"object": args[0], "target": args[1]}
                )
            elif name in ("open", "close", "turnon", "turnoff") and args:
                expected = {
                    "open": "Open", "close": "Close",
                    "turnon": "TurnOn", "turnoff": "TurnOff",
                }[name]
                expected_args = {"target": args[0]}

        scores = {}
        for candidate in candidates:
            name = candidate["name"]
            args = candidate.get("args", {})
            matches = name == expected and all(
                args.get(key) == value for key, value in expected_args.items()
            )
            if name == "Done":
                value = 0.99 if belief.all_goals_satisfied() else 0.01
                rationale = (
                    "All BDDL goal predicates are satisfied."
                    if value > 0.5 else "The task is not complete."
                )
            elif matches:
                value = 0.98
                rationale = "This skill advances the earliest unsatisfied BDDL goal."
            else:
                value = 0.25
                rationale = "Possible alternative, but not the immediate goal action."
            scores[candidate["id"]] = {"relevance": value, "rationale": rationale}
        return scores
