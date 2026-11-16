Analyse **{title}** from the sources below, then write a concise, spoiler-free listener summary.

Purpose: {purpose}
Audience: {audience}
Target size for `summary`: {target_size}

{inputs_text}

Return JSON with:
- `premise`: 1-sentence spoiler-free overview of what the book is about.
- `liked`: key strengths praised by listeners/reviewers (grounded in sources).
- `disliked`: key criticisms or caveats noted by listeners/reviewers (or "none noted").
- `summary`: one polished, readable, spoiler-free paragraph within `{target_size}` combining premise, likes, and dislikes with complete sentences and no preamble.
