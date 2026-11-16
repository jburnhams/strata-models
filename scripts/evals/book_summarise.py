"""Build the eval dataset `book-summarise-8`: eight audiobooks with synopses and mixed reader reviews, to test local and cloud models on role `summarise`. Run once:

.venv/bin/python scripts/evals/book_summarise.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels.evals import core as C

BOOKS = [
    (
        "brave-new-world",
        "Brave New World",
        [
            {
                "source": "synopsis",
                "text": "Aldous Huxley's dystopian classic imagines a future World State where citizens are genetically engineered into castes, conditioned from birth, and pacified with the drug soma until an outsider from a Savage Reservation challenges the system.",
            },
            {
                "source": "review-1",
                "text": "Eerily prescient about consumerism and distraction; the world-building is sharp and Michael York's narration fits the clinical tone.",
            },
            {
                "source": "review-2",
                "text": "The ideas are brilliant, though the characters feel more like mouthpieces for philosophy than real people, and the middle sags.",
            },
            {
                "source": "review-3",
                "text": "A bit dated in places and heavy on dialogue, but the ending still hits hard.",
            },
        ],
    ),
    (
        "educating-daisy",
        "Educating Daisy with Daisy May Cooper",
        [
            {
                "source": "synopsis",
                "text": "Six celebrities face the challenge of convincing comedian Daisy May Cooper to read a book, pitching their favourites with plenty of banter and digressions.",
            },
            {
                "source": "review-1",
                "text": "Laugh-out-loud funny; Daisy's chemistry with her guests makes it feel like eavesdropping on friends at the pub.",
            },
            {
                "source": "review-2",
                "text": "Very entertaining if you already like her humour, though listeners wanting serious literary discussion or structured book reviews may find it too chaotic.",
            },
        ],
    ),
    (
        "david-copperfield",
        "David Copperfield",
        [
            {
                "source": "synopsis",
                "text": "Charles Dickens's semi-autobiographical novel follows David Copperfield from a harsh childhood under his cruel stepfather to his struggles and growth as a young writer in Victorian London.",
            },
            {
                "source": "review-1",
                "text": "Richard Armitage's narration brings every eccentric Dickens character vividly to life; warm, funny, and deeply moving.",
            },
            {
                "source": "review-2",
                "text": "At over thirty-six hours it is a massive commitment with occasional melodrama and Victorian sentimentality, yet it rewards patience.",
            },
        ],
    ),
    (
        "project-hail-mary",
        "Project Hail Mary",
        [
            {
                "source": "synopsis",
                "text": "A lone astronaut wakes up on a spacecraft with amnesia, gradually piecing together his mission to save Earth from an extinction-level solar dimming event.",
            },
            {
                "source": "review-1",
                "text": "Ray Porter's audio performance is phenomenal—especially how the alien communication is handled in audio. Fast-paced problem-solving sci-fi.",
            },
            {
                "source": "review-2",
                "text": "The protagonist's 'dad-joke' quips and overly convenient science breakthroughs grated on me, even though the plot kept me listening.",
            },
        ],
    ),
    (
        "the-secret-history",
        "The Secret History",
        [
            {
                "source": "synopsis",
                "text": "At an elite New England college, a tight-knit group of classics students under a charismatic professor cross from academic obsession into murder and live with the fallout.",
            },
            {
                "source": "review-1",
                "text": "Hypnotic atmosphere and razor-sharp prose; Donna Tartt's own narration gives it a haunting, intimate quality.",
            },
            {
                "source": "review-2",
                "text": "Every character is deeply unlikable and the pacing slows to a crawl in the second half, which makes the pretension hard to stomach for some.",
            },
        ],
    ),
    (
        "empire-of-pain",
        "Empire of Pain",
        [
            {
                "source": "synopsis",
                "text": "Patrick Radden Keefe traces three generations of the Sackler family, from their rise in medical advertising and art philanthropy to Purdue Pharma's role in the opioid crisis.",
            },
            {
                "source": "review-1",
                "text": "Meticulously researched and reads like a thriller; the author's narration is clear, measured, and gripping.",
            },
            {
                "source": "review-2",
                "text": "Exhaustive and infuriating—a few listeners felt the early chapters on family history dragged before OxyContin enters the picture.",
            },
        ],
    ),
    (
        "thursday-murder-club",
        "The Thursday Murder Club",
        [
            {
                "source": "synopsis",
                "text": "Four retirees in a peaceful British village meet weekly to investigate unsolved killings, until a local developer turns up dead and they have a live case.",
            },
            {
                "source": "review-1",
                "text": "Charming, witty, and cosy with genuinely endearing older protagonists and crisp British narration by Lesley Manville.",
            },
            {
                "source": "review-2",
                "text": "Too many subplots and a contrived mystery resolution; it's more about the banter than tight detective plotting.",
            },
        ],
    ),
    (
        "dune",
        "Dune",
        [
            {
                "source": "synopsis",
                "text": "Young Paul Atreides must survive betrayal and political intrigue on the desert planet Arrakis, the sole source of the universe's most valuable substance.",
            },
            {
                "source": "review-1",
                "text": "Monumental world-building, ecology, and political depth that defined modern science fiction.",
            },
            {
                "source": "review-2",
                "text": "Dense terminology with a slow opening, and the audiobook's inconsistent mix of full-cast and solo narration chapters annoys many listeners.",
            },
        ],
    ),
]


def main():
    items = [
        dict(
            id=bid,
            input=dict(
                preset="summarise-v1",
                context=dict(
                    title=title,
                    purpose="Help a listener decide whether to pick this audiobook next: what it is about and what people like and dislike about it.",
                    audience="An audiobook listener browsing their personal library.",
                    target_size="One concise paragraph (3-4 sentences).",
                    inputs=inputs,
                ),
                max_tokens=400,
                temperature=0.2,
            ),
            tags=dict(book=bid),
        )
        for bid, title, inputs in BOOKS
    ]
    ds = C.write_dataset(
        "book-summarise-8",
        items,
        description="Eight audiobooks with synopses and mixed reader reviews for summarisation evals",
    )
    print(f"{len(ds.items)} books, version {ds.version}")


if __name__ == "__main__":
    main()
