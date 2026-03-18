from __future__ import annotations


class GeneratorModule:
    def generate(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I don't have enough evidence from reviews to answer this question."
        return "\n".join(contexts[:1])

