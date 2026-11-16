"""Build the eval dataset `book-summarise-real-10` from real enriched Library books in `strata-listory`'s `listory` Postgres schema.

Run once (with `STRATA_DB_LIVE=1 STRATA_DB_DATABASE=omarchy`):

    STRATA_DB_LIVE=1 STRATA_DB_DATABASE=omarchy .venv/bin/python scripts/evals/book_summarise_real.py"""

import os
import sys

from sqlalchemy import text
from strata_python.db import connect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels.evals import core as C

PRIORITY = (
    "hardcover",
    "audible_local",
    "audible",
    "audnexus",
    "sidecar",
    "openlibrary_local",
    "openlibrary",
    "googlebooks",
    "applebooks",
)


def main():
    eng = connect("listory")
    items = []
    with eng.connect() as c:
        runs = c.execute(
            text(
                """
                SELECT DISTINCT ON (r.listen_id) r.id, r.listen_id, l.label
                FROM enrichment_run r
                JOIN listen l ON l.id = r.listen_id
                WHERE r.finished_at IS NOT NULL
                ORDER BY r.listen_id, r.id DESC
                """
            )
        ).fetchall()
        for run_id, listen_id, label in runs[:10]:
            rows = c.execute(
                text(
                    """
                    SELECT source, synopsis, reviews
                    FROM enrichment_result
                    WHERE run_id = :rid
                    ORDER BY id
                    """
                ),
                {"rid": run_id},
            ).fetchall()
            inputs = []
            seen_syn = set()
            by_src = {}
            for src, syn, revs in rows:
                by_src.setdefault(src, []).append((syn, revs or []))
            for src in PRIORITY:
                for syn, _ in by_src.get(src, []):
                    s = (syn or "").strip()
                    if not s:
                        continue
                    key = " ".join(s.lower().split())[:200]
                    if key in seen_syn:
                        continue
                    seen_syn.add(key)
                    inputs.append({"source": f"{src}:synopsis", "text": s[:1500]})
                    break
            reviews = []
            for src, _, revs in rows:
                for rev in revs or []:
                    txt = (rev.get("text") or "").strip()
                    if txt:
                        reviews.append(
                            {
                                "source": f"{src}:review",
                                "rating": rev.get("rating"),
                                "text": txt[:800],
                            }
                        )
            inputs.extend(reviews[:8])
            if not inputs:
                continue
            items.append(
                dict(
                    id=f"listen-{listen_id}",
                    input=dict(
                        preset="summarise-v1",
                        context=dict(
                            title=label,
                            purpose="Help a listener decide whether to pick this audiobook next: what it is about and what people like and dislike about it.",
                            audience="An audiobook listener browsing their personal library.",
                            target_size="One concise paragraph (3-4 sentences).",
                            inputs=inputs,
                        ),
                        max_tokens=400,
                        temperature=0.2,
                    ),
                    tags=dict(listen_id=listen_id, label=label),
                )
            )
    ds = C.write_dataset(
        "book-summarise-real-10",
        items,
        description="Real enriched Library books from strata-listory (synopses + reviews from Hardcover, Audible, Open Library, Apple Books)",
    )
    print(f"{len(ds.items)} real books, version {ds.version}")


if __name__ == "__main__":
    main()
