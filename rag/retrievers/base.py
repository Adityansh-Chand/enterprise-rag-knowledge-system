"""The retriever contract every implementation satisfies.

One interface, several genuinely different ranking strategies, one evaluation
harness. Adding a method means implementing `index` and `search` -- nothing in
the harness or the service changes.
"""
from typing import Protocol, Sequence


class Retriever(Protocol):
    """Ranks indexed documents against a query.

    `search` returns (document_index, score) pairs in descending score order,
    where document_index refers to the position in the sequence passed to
    `index`. Scores are only meaningful within a single retriever -- they are
    not comparable across implementations, which is why fusion works on ranks.
    """

    name: str

    def index(self, documents: Sequence[str]) -> None:
        ...

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        ...
