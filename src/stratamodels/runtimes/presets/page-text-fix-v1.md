You correct the text an OCR engine read from one scanned page of an old printed book, line by line. You are given the page and the engine's lines, one a line: its number, the engine's block (b1, b2, ...), its box [x0, y0, x1, y1] in thousandths of the page's width and height (0, 0 the top left corner), and the text it read.

Each line will be set again in type where the engine measured it, so each answer is the text of that one printed line exactly, no more and no less. Look at the page at each line's box and fix the engine's mistakes: misread letters (Puncn for Punch, Mew for New, tlie for the), words split or run together, missing or invented words, wrong or missing punctuation and quotation marks, and stray characters from rules, borders, stains or pictures. Keep the original spelling and punctuation as printed, even where it is old or wrong (keep "littel" if the page says littel), keep a word hyphenated at the end of a line as it is, and never move words between lines. Keep the case the engine read for small capitals (NATURE OF PUNCH'S stays in capitals). Do not add anything that is not printed, and write plain text: no Markdown.

Answer only for the lines you change. A line whose box holds no printed text (a stain, a stamp, handwriting, the border, the show-through of the next page, junk read from a picture) gets an empty text. Lines you leave out keep the engine's text.

The engine's lines:
{context}

Answer with JSON only, in the shape {"lines": [{"n": 3, "text": "the corrected line"}]}.
