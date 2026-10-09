"""Minimal stubs for tiktoken (runtime package ships without ``py.typed``)."""

from typing import Protocol

class Encoding(Protocol):
    def encode(self, text: str) -> list[int]: ...

def get_encoding(encoding_name: str) -> Encoding: ...
