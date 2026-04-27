"""GSSC context builder for the PaperNote harness agent.

The builder follows the Gather-Select-Structure-Compress pipeline described in
the context-engineering notes. It intentionally stays deterministic so the RAG
pipeline still works when no extra LLM key is available for compression.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


@dataclass
class ContextPacket:
    """Candidate information unit used by the GSSC pipeline."""

    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.content = str(self.content or "").strip()
        self.relevance_score = max(0.0, min(1.0, float(self.relevance_score)))
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class ContextConfig:
    """Context construction knobs."""

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7
    history_limit: int = 6

    def __post_init__(self):
        if not 0.0 <= self.reserve_ratio <= 1.0:
            raise ValueError("reserve_ratio must be in [0, 1]")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be in [0, 1]")
        total = self.recency_weight + self.relevance_weight
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("recency_weight + relevance_weight must equal 1.0")
        self.max_tokens = max(256, int(self.max_tokens))
        self.history_limit = max(0, int(self.history_limit))


@dataclass
class BuiltContext:
    """Structured context and diagnostics returned by ContextBuilder."""

    context: str
    selected_packets: list[ContextPacket]
    stats: dict[str, Any]


class ContextBuilder:
    """Build high-signal model context with GSSC."""

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def build(
        self,
        *,
        user_query: str,
        conversation_history: Optional[list[Any]] = None,
        system_instructions: str = "",
        state_summary: str = "",
        tool_results: Optional[list[dict[str, Any]]] = None,
        custom_packets: Optional[list[ContextPacket]] = None,
        output_instructions: str = "",
    ) -> BuiltContext:
        """Run Gather-Select-Structure-Compress and return a structured prompt."""

        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            state_summary=state_summary,
            tool_results=tool_results or [],
            custom_packets=custom_packets or [],
            output_instructions=output_instructions,
        )
        selected = self._select(packets, user_query)
        structured = self._structure(selected, user_query)
        compressed = False
        if self.config.enable_compression:
            before_tokens = self.count_tokens(structured)
            structured = self._compress(structured, self.config.max_tokens)
            compressed = self.count_tokens(structured) < before_tokens

        stats = {
            "gathered_packets": len(packets),
            "selected_packets": len(selected),
            "token_count": self.count_tokens(structured),
            "max_tokens": self.config.max_tokens,
            "compressed": compressed,
            "sections": self._section_counts(selected),
        }
        return BuiltContext(context=structured, selected_packets=selected, stats=stats)

    def _gather(
        self,
        *,
        user_query: str,
        conversation_history: list[Any],
        system_instructions: str,
        state_summary: str,
        tool_results: list[dict[str, Any]],
        custom_packets: list[ContextPacket],
        output_instructions: str,
    ) -> list[ContextPacket]:
        now = self._now()
        packets: list[ContextPacket] = []

        if system_instructions.strip():
            packets.append(
                self.packet(
                    system_instructions,
                    relevance_score=1.0,
                    metadata={"type": "system_instruction", "priority": "high"},
                    timestamp=now,
                )
            )

        packets.append(
            self.packet(
                f"User query: {user_query}",
                relevance_score=1.0,
                metadata={"type": "task", "priority": "high"},
                timestamp=now,
            )
        )

        if state_summary.strip():
            packets.append(
                self.packet(
                    state_summary,
                    relevance_score=0.9,
                    metadata={"type": "state", "priority": "high"},
                    timestamp=now,
                )
            )

        for result in tool_results:
            content = str(result.get("content") or "").strip()
            if not content:
                continue
            packets.append(
                self.packet(
                    content,
                    relevance_score=self._tool_relevance(result),
                    metadata={
                        "type": self._tool_packet_type(result),
                        "tool": result.get("tool", ""),
                        "status": result.get("status", "ok"),
                    },
                    timestamp=self._parse_time(result.get("timestamp")) or now,
                )
            )

        for message in conversation_history[-self.config.history_limit :]:
            text = self._message_text(message)
            if not text:
                continue
            role = self._message_role(message)
            packets.append(
                self.packet(
                    f"{role}: {text}",
                    relevance_score=0.55,
                    metadata={"type": "conversation_history", "role": role},
                    timestamp=now,
                )
            )

        packets.extend(packet for packet in custom_packets if packet.content)

        if output_instructions.strip():
            packets.append(
                self.packet(
                    output_instructions,
                    relevance_score=1.0,
                    metadata={"type": "output_instruction", "priority": "high"},
                    timestamp=now,
                )
            )

        return packets

    def _select(self, packets: list[ContextPacket], user_query: str) -> list[ContextPacket]:
        high_priority = [
            packet
            for packet in packets
            if packet.metadata.get("priority") == "high"
            or packet.metadata.get("type") in {"system_instruction", "task", "output_instruction"}
        ]
        others = [packet for packet in packets if packet not in high_priority]

        selected: list[ContextPacket] = []
        current_tokens = 0
        for packet in high_priority:
            selected.append(packet)
            current_tokens += packet.token_count

        available_tokens = self.config.max_tokens
        if current_tokens >= available_tokens:
            return selected

        scored: list[tuple[float, ContextPacket]] = []
        for packet in others:
            relevance = packet.relevance_score
            if math.isclose(relevance, 0.5, abs_tol=1e-9):
                relevance = self._calculate_relevance(packet.content, user_query)
                packet.relevance_score = relevance
            if relevance < self.config.min_relevance:
                continue
            recency = self._calculate_recency(packet.timestamp)
            score = (
                self.config.relevance_weight * relevance
                + self.config.recency_weight * recency
            )
            scored.append((score, packet))

        scored.sort(key=lambda item: item[0], reverse=True)
        for _, packet in scored:
            if current_tokens + packet.token_count > available_tokens:
                continue
            selected.append(packet)
            current_tokens += packet.token_count

        return selected

    def _structure(self, selected_packets: list[ContextPacket], user_query: str) -> str:
        groups: dict[str, list[str]] = {
            "system_instruction": [],
            "task": [],
            "state": [],
            "evidence": [],
            "context": [],
            "output_instruction": [],
        }

        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")
            if packet_type == "system_instruction":
                groups["system_instruction"].append(packet.content)
            elif packet_type == "task":
                groups["task"].append(packet.content)
            elif packet_type == "state":
                groups["state"].append(packet.content)
            elif packet_type in {"rag_result", "research_result"}:
                groups["evidence"].append(packet.content)
            elif packet_type == "output_instruction":
                groups["output_instruction"].append(packet.content)
            else:
                groups["context"].append(packet.content)

        sections: list[str] = []
        if groups["system_instruction"]:
            sections.append("[Role & Policies]\n" + "\n".join(groups["system_instruction"]))
        task_text = "\n".join(groups["task"]) or f"User query: {user_query}"
        sections.append("[Task]\n" + task_text)
        if groups["state"]:
            sections.append("[State]\n" + "\n".join(groups["state"]))
        if groups["evidence"]:
            sections.append("[Evidence]\n" + "\n\n---\n\n".join(groups["evidence"]))
        if groups["context"]:
            sections.append("[Context]\n" + "\n\n".join(groups["context"]))
        output_text = "\n".join(groups["output_instruction"]) or (
            "Answer accurately using the supplied context. Cite paper evidence as [S#], "
            "Tavily Research evidence as [R#], and notes as [N#] when used."
        )
        sections.append("[Output]\n" + output_text)
        return "\n\n".join(sections).strip()

    def _compress(self, context: str, max_tokens: int) -> str:
        current_tokens = self.count_tokens(context)
        if current_tokens <= max_tokens:
            return context

        sections = re.split(r"\n(?=\[[A-Za-z &]+\]\n)", context)
        compressed_sections: list[str] = []
        used_tokens = 0
        for section in sections:
            section_tokens = self.count_tokens(section)
            if used_tokens + section_tokens <= max_tokens:
                compressed_sections.append(section)
                used_tokens += section_tokens
                continue
            remaining = max_tokens - used_tokens
            if remaining <= 40:
                break
            compressed_sections.append(
                self._truncate_text(section, remaining) + "\n[... context compressed ...]"
            )
            break
        return "\n".join(compressed_sections).strip()

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if self.count_tokens(text) <= max_tokens:
            return text
        ratio = max_tokens / max(self.count_tokens(text), 1)
        max_chars = max(200, int(len(text) * ratio))
        return text[:max_chars].rstrip()

    def packet(
        self,
        content: str,
        *,
        relevance_score: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ContextPacket:
        return ContextPacket(
            content=content,
            timestamp=timestamp or self._now(),
            token_count=self.count_tokens(content),
            relevance_score=relevance_score,
            metadata=metadata or {},
        )

    @classmethod
    def count_tokens(cls, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        cjk = re.findall(r"[\u3400-\u9fff]", text)
        words = re.findall(r"[A-Za-z0-9_]+", text)
        remaining_chars = max(len(text) - len(cjk) - sum(len(word) for word in words), 0)
        return max(1, len(cjk) + len(words) + remaining_chars // 4)

    def _calculate_relevance(self, content: str, query: str) -> float:
        query_terms = set(self._terms(query))
        if not query_terms:
            return 0.0
        content_terms = set(self._terms(content))
        if not content_terms:
            return 0.0
        overlap = query_terms & content_terms
        return len(overlap) / max(len(query_terms), 1)

    def _calculate_recency(self, timestamp: datetime) -> float:
        age_hours = max((self._now() - timestamp).total_seconds() / 3600, 0.0)
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24)))

    def _terms(self, text: str) -> list[str]:
        normalized = str(text or "").lower()
        words = re.findall(r"[a-z0-9_]{2,}", normalized)
        cjk = re.findall(r"[\u3400-\u9fff]", normalized)
        cjk_bigrams = [a + b for a, b in zip(cjk, cjk[1:])]
        return words + cjk + cjk_bigrams

    def _section_counts(self, packets: Iterable[ContextPacket]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for packet in packets:
            packet_type = str(packet.metadata.get("type", "general"))
            counts[packet_type] = counts.get(packet_type, 0) + 1
        return counts

    def _tool_relevance(self, result: dict[str, Any]) -> float:
        tool = result.get("tool")
        status = result.get("status", "ok")
        if status != "ok":
            return 0.15
        if tool == "paper_rag":
            return 0.9
        if tool == "tavily_research":
            return 0.75
        if tool == "note_tool":
            return 0.7
        return 0.5

    def _tool_packet_type(self, result: dict[str, Any]) -> str:
        tool = result.get("tool")
        if tool == "paper_rag":
            return "rag_result"
        if tool == "tavily_research":
            return "research_result"
        if tool == "note_tool":
            return "note"
        return "general"

    def _message_text(self, message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(part for part in text_parts if part).strip()
        return str(content or "").strip()

    def _message_role(self, message: Any) -> str:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
        return str(role or "message")

    def _parse_time(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
