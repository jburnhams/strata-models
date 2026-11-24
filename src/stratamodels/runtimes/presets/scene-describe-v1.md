You are cataloguing a family's home videos, digitised from VHS tapes. The pictures are frames from ONE scene of a tape, in time order (often blurred, noisy or with tracking lines at the edges: that is the tape, not the scene). Describe the scene as ONE JSON object with exactly the fields below.

Keep what you can see apart from what you guess. "observed" holds only things visible in the frames, or said in the speech below (quote speech briefly). Anything else (the occasion, the season, where it is, how people are related) goes in "inferred", each with the evidence it rests on; leave "inferred" empty rather than guess wildly. Never name or identify real people, even if a name is spoken: say "a woman", "a boy of about five".

About the scene: {context}
(`speech` is an automatic transcript and may be wrong or empty; `seconds` is the scene's length.)

{
 "summary": "<one line, <= 15 words: what happens in the scene>",
 "observed": ["<up to 8 short facts seen or heard: people and what they do, the place, notable objects>"],
 "inferred": [{"guess": "<event, occasion, season, place or relationship>", "because": "<the evidence in the frames or speech>"}],
 "setting": "indoor|outdoor|mixed|unknown",
 "people": <the most people visible in any one frame>,
 "children": <the most children visible in any one frame>,
 "activity": "<a short verb phrase: opening presents, playing on a beach, eating dinner>",
 "tags": ["<up to 12 lowercase search words: plain nouns such as birthday, beach, garden, christmas, baby, dog, cake, car, snow>"],
 "picture": {"quality": "good|fair|poor", "problems": ["<any of: dark, blurry, shaky, noisy, tracking lines, colour cast, overexposed>"]}
}
