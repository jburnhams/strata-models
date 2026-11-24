You are cataloguing a family's home videos, digitised from camcorder tapes of the 1980s and 1990s. The pictures are frames from ONE scene of a tape, in time order, a few seconds apart: the camera moves, so they may show different parts of one room, and different people. Describe the whole scene, not any one frame, as ONE JSON object with exactly the fields below.

Keep what you can see apart from what you guess. "observed" holds only things visible in the frames, or said in the speech below (quote speech briefly). Anything else (the occasion, the season, where it is, how people are related) goes in "inferred", each with the evidence it rests on; leave "inferred" empty rather than guess wildly. Never name or identify real people, even if a name is spoken: say "a woman", "a boy of about five".

About the scene: {context}
(`speech` is an automatic transcript, often wrong, and may be empty: trust the frames over it, and do not explain a picture by words that do not fit it. `seconds` is the scene's length.)

{
 "summary": "<one line, <= 15 words: what happens in the scene>",
 "observed": ["<up to 8 short facts seen or heard: people and what they do, the place, notable objects>"],
 "inferred": [{"guess": "<event, occasion, season, place or relationship>", "because": "<the evidence in the frames or speech>"}],
 "setting": "indoor|outdoor|mixed|unknown",
 "people": <the most real people visible in any one frame (people in photographs, drawings or on cards do not count)>,
 "children": <the most children visible in any one frame>,
 "activity": "<a short verb phrase: opening presents, playing on a beach, eating dinner>",
 "tags": ["<up to 12 lowercase search words: plain nouns such as birthday, beach, garden, christmas, baby, dog, cake, car, snow>"],
 "picture": {"quality": "good|fair|poor", "problems": ["<from the list below, usually none or one>"]}
}

Judge "picture" against other home videos of its time, not against a modern camera: soft detail, faint grain, dull colour and a thin torn or black line along the very edge of the frame are how every tape looks and are not problems. "good": everything that matters can be seen clearly for a tape. "fair": something gets in the way. "poor": it is hard to make out what is going on. List a problem only when it spoils the frames beyond that: "dark" (faces or actions are lost in shadow), "blurry" (out of focus or smeared so that faces cannot be made out), "shaky" (the camera jerks about), "noisy" (heavy speckle over the whole picture), "tracking lines" (bands of white noise or torn picture across the MIDDLE of the frame, not at its edge), "colour cast" (the whole picture is plainly tinted), "overexposed" (large areas burnt out to white).
