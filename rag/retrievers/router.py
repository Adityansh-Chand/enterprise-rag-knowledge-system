"""Route each query to the retriever most likely to answer it.

Weighted fusion is already implemented, and tuning it chose "pure dense" on both
corpora -- a single global weight of 1.0 for the semantic side and 0.0 for the
lexical one. That is an honest result, and it is also a limitation: one weight
applied to every query cannot exploit the fact that BM25 is genuinely better on
*some* queries.

The synthetic corpus was built to make that visible. `ERR-4021 remediation steps`
is a lexical problem -- the answer contains that exact token and nothing else
does, so an embedding that maps it near other error codes actively hurts.
`customers cannot complete purchases at the final step` is the opposite: it shares
no content word with the document about checkout latency, and BM25 scores it near
zero.

So the routing decision is made **per query, from the query text alone**:

    a query containing an identifier-shaped token  ->  lexical
    everything else                                ->  semantic

Two constraints this is written under, both of which make the result weaker and
believable:

**The rule may not look at the corpus's query-type labels.** Those labels exist in
`datasets/queries.json`, and routing on them would score beautifully and mean
nothing -- it would be the circular evaluation this repository was rebuilt to
remove. The rule sees the same string a user would type.

**The rule is fixed before measurement, not tuned against the report split.**
`training/tune_router.py` selects on a dev half and reports on the other, the same
discipline `training/tune_fusion.py` uses.
"""
import re

# Identifier-shaped tokens: error codes (ERR-4021), dotted config keys
# (retry.max_attempts), API paths (/v2/invoices), versions (1.4), clause numbers.
# Deliberately narrow -- a rule that fires on ordinary prose would route
# everything to BM25 and simply reinvent lexical-only retrieval.
IDENTIFIER = re.compile(
    r"(?:\b[A-Z]{2,}-\d+\b"          # ERR-4021, RFC-2119
    r"|\b\w+\.\w+(?:\.\w+)*\b"       # retry.max_attempts
    r"|/v\d+/\w+"                    # /v2/invoices
    r"|\bS\d+\.\d+\b"                # section numbers
    r"|§\s*\d+(?:\.\d+)?)"      # 7.3
)


def looks_lexical(query):
    """True when the query carries a token that must match exactly.

    Uppercase acronyms alone are deliberately NOT enough: "what does our DPA say
    about sub-processors" is prose that happens to contain an acronym, and the
    document answering it uses the expansion. Routing that to BM25 loses.
    """
    return bool(IDENTIFIER.search(query))


class RouterRetriever:
    """Delegates each query to `lexical` or `semantic` by a text-only rule."""

    def __init__(self, lexical, semantic, rule=looks_lexical):
        self.lexical = lexical
        self.semantic = semantic
        self.rule = rule
        self.name = f"router({lexical.name}|{semantic.name})"
        self.routed = {"lexical": 0, "semantic": 0}

    def index(self, documents):
        # Both are indexed: routing decides per query, so both must be ready.
        # This is the cost of routing, and it is honest to state it -- the memory
        # and index time are the sum of both, not a saving.
        self.lexical.index(documents)
        self.semantic.index(documents)

    def search(self, query, k):
        if self.rule(query):
            self.routed["lexical"] += 1
            return self.lexical.search(query, k)
        self.routed["semantic"] += 1
        return self.semantic.search(query, k)
