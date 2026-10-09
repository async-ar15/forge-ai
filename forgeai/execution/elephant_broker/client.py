"""v1 ElephantBroker stub — interface only; production logic ships in v2."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

from forgeai.execution.elephant_broker.types import GroundedPrompt, MemoryContext

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class ElephantBrokerClient:
    """Minimal broker surface for execution; DEBUG pass-through in v1.

    This is a v1 stub — full ElephantBroker implementation is v2.
    """

    def ground_prompt(
        self,
        prompt_utf8: str,
        retrieved_context: Sequence[str],
    ) -> GroundedPrompt:
        """Return ``prompt_utf8`` unchanged; logs shape only."""

        _LOG.debug(
            "elephant_broker_v1_stub ground_prompt chars=%s ctx_items=%s",
            len(prompt_utf8),
            len(retrieved_context),
        )
        return GroundedPrompt(text_utf8=prompt_utf8)

    def record_tool_call(
        self,
        tool_name: str,
        inputs_json_utf8: str,
        outputs_json_utf8: str,
    ) -> None:
        """No-op audit sink in v1."""

        _LOG.debug(
            "elephant_broker_v1_stub record_tool_call name=%s in=%s out=%s",
            tool_name,
            len(inputs_json_utf8),
            len(outputs_json_utf8),
        )

    def get_memory_context(self, tenant_id: str, session_id: str) -> MemoryContext:
        """Return empty memory; v2 will hydrate from durable store."""

        _LOG.debug(
            "elephant_broker_v1_stub get_memory_context tenant=%s session=%s",
            tenant_id,
            session_id,
        )
        return MemoryContext(snippets_utf8=())


__all__ = ["ElephantBrokerClient"]
