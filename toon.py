import json


class Toon:
    def parse_json(self, text: str) -> dict | None:
        cleaned = text.strip()
        direct = self._safe_loads(cleaned)
        if direct is not None:
            return direct
        extracted = self._extract_json_block(cleaned)
        return self._safe_loads(extracted) if extracted else None

    def _safe_loads(self, raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_json_block(self, text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        return text[start : end + 1]
