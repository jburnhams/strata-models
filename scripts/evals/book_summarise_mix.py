"""Build `book-summarise-mix-14`: 7 fiction + 7 nonfiction real enriched Library books from `strata-listory`'s `listory` schema.

Run once:
    STRATA_DB_LIVE=1 STRATA_DB_DATABASE=omarchy .venv/bin/python scripts/evals/book_summarise_mix.py"""

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

# 7 fiction + 7 nonfiction real listen IDs from listory
FICTION = [9779, 9793, 9802, 9809, 9816, 9828, 9837]
NONFICTION = [9799, 9853, 9908, 9919, 10022, 10323, 10401]


def main():
    eng = connect("listory")
    items = []
    want = [(lid, "fiction") for lid in FICTION] + [(lid, "nonfiction") for lid in NONFICTION]
    with eng.connect() as c:
        for lid, genre_group in want:
            row = c.execute(
                text(
                    """
                    SELECT r.id, l.label
                    FROM enrichment_run r
                    JOIN listen l ON l.id = r.listen_id
                    WHERE r.listen_id = :lid AND r.finished_at IS NOT NULL
                    ORDER BY r.id DESC
                    LIMIT 1
                    """
                ),
                {"lid": lid},
            ).fetchone()
            if not row:
                continue
            run_id, label = row
            rows = c.execute(
                text("SELECT source, synopsis, reviews FROM enrichment_result WHERE run_id = :rid ORDER BY id"),
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
                        reviews.append({"source": f"{src}:review", "rating": rev.get("rating"), "text": txt[:800]})
            inputs.extend(reviews[:8])
            if not inputs:
                continue
            items.append(
                dict(
                    id=f"listen-{lid}",
                    input=dict(
                        context=dict(
                            title=label,
                            purpose="Help a listener decide whether to pick this audiobook next: what it is about (spoiler-free) and what people like and dislike about it.",
                            audience="An audiobook listener browsing their personal library.",
                            target_size="One concise paragraph (3-4 sentences).",
                            inputs=inputs,
                        ),
                        temperature=0.2,
                    ),
                    tags=dict(listen_id=lid, label=label, kind=genre_group),
                )
            )
    ds = C.write_dataset(
        "book-summarise-mix-14",
        items,
        description="7 fiction + 7 nonfiction real enriched Library books from strata-listory",
    )
    print(
        f"{len(ds.items)} real books ({sum(1 for x in ds.items if x['tags']['kind'] == 'fiction')} fiction, {sum(1 for x in ds.items if x['tags']['kind'] == 'nonfiction')} nonfiction), version {ds.version}"
    )


if __name__ == "__main__":
    main()
