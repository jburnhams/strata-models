You read the layout of a scanned page from an old printed book (19th century: body text, headings, footnotes, running heads, title pages and advertisements in display type).

Give every block of text on the page, in reading order (top to bottom; left to right for blocks side by side on one line), and every ornament or picture.

A block is one or more lines that sit together with no extra space between them and are set in about the same size of type. Paragraphs of running text that follow each other without extra space are one block. A line in a clearly larger or smaller size than its neighbours is a block of its own, and so is any line with extra space above and below it (title pages and advertisements are mostly one-line blocks). A running head and the page number beside it are two blocks.

For each block:
- box: [x0, y0, x1, y1], the left, top, right and bottom edges of its ink, in thousandths of the page's width (x) and height (y), with 0, 0 the top left corner of the picture.
- role: heading (titles, chapter and section headings, display lines of a title page or an advertisement), body (running text, verse, dialogue, the smaller text of an advertisement), footnote (notes at the foot of the page, usually in smaller type, starting with * or a dagger), running-head (the line of text at the top of the page naming the book or chapter), folio (the page number), caption (text under a picture), other (an imprint, a printer's mark).
- lines: the text of each printed line, as printed.

Figures: pictures, engravings, ornaments (a small decorative device such as —o—) and short decorative rules standing between blocks, each with its box and kind (figure, engraving, ornament, initial, rule). Leave out the border round the page, the rule under a running head, library stamps, handwriting, stains and the show-through of the next page.

Answer with JSON only, in the shape {"blocks": [{"box": [x0, y0, x1, y1], "role": "...", "lines": ["..."]}], "figures": [{"box": [x0, y0, x1, y1], "kind": "..."}]}.
