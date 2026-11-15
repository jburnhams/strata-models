You correct an OCR engine's reading of the layout of a scanned page from an old printed book (19th century: body text, headings, footnotes, running heads, title pages and advertisements in display type). You are given the page and the engine's lines, one a line: its number, the engine's block (b1, b2, ...), its box [x0, y0, x1, y1] in thousandths of the page's width and height (0, 0 the top left corner), and the text it read.

The engine's lines are measured well but grouped badly: it splits a paragraph, puts a heading in with the text under it, and reads stains, stamps and the show-through of the next page as lines of junk. Look at the page and group the lines into the page's real blocks, in reading order (top to bottom; left to right for blocks side by side on one line).

A block is one or more lines that sit together with no extra space between them and are set in about the same size of type. Paragraphs of running text that follow each other without extra space are one block. A line in a clearly larger or smaller size than its neighbours is a block of its own, and so is any line with extra space above and below it (title pages and advertisements are mostly one-line blocks). A running head and the page number beside it are two blocks (the engine's line is already split where words stand far apart).

Split only where the print does: by space between lines or a change of type size, never by meaning. A block is set apart by eye, not by what it says, so keep together:
- the speeches of a play or a dialogue that follow each other without extra space, even where the speaker changes: one block;
- the notes at the foot of a page, all one footnote block when no extra space parts them, even where a new note starts;
- the lines of a byline, an imprint or an address set together in one size of type: one block.
When in doubt whether two runs of lines are one block or two, they are one.

Leave a line out only when no printed text stands at its box. A line the engine read badly (a smudged running head, garbled letters) is still printed text: keep it in its block.

For each block:
- lines: the numbers of its lines. Use each number once. Leave out lines that are not printed text of the page (stains, stamps, handwriting, the border, show-through, junk read from a picture).
- role: heading (titles, chapter and section headings, display lines of a title page or an advertisement), body (running text, verse, dialogue, the smaller text of an advertisement), footnote (notes at the foot of the page, usually in smaller type, starting with * or a dagger), running-head (the line of text at the top of the page naming the book or chapter), folio (the page number), caption (text under a picture), other (an imprint, a printer's mark).

Figures: pictures, engravings, ornaments (a small decorative device such as —o—) and short decorative rules standing between blocks, each with its box [x0, y0, x1, y1] in thousandths and kind (figure, engraving, ornament, initial, rule). The engine reads none of these. Leave out the border round the page, the rule under a running head, library stamps, handwriting, stains and the show-through of the next page.

The engine's lines:
{context}

Answer with JSON only, in the shape {"blocks": [{"lines": [1, 2], "role": "..."}], "figures": [{"box": [x0, y0, x1, y1], "kind": "..."}]}.
