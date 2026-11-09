You are cataloguing a personal photo archive. The image is the camera's own JPEG of the shot (its in-camera rendering, before any editing). Describe it as ONE JSON object with exactly the fields below. Use the capture metadata as context (do not just copy it back). Be factual: describe what is visible, say "unknown" rather than guess wildly, and never identify real people by name.

Capture metadata: {context}

{
 "caption": "<one sentence, <= 20 words, what the photo shows>",
 "description": "<3-5 sentences: subject, setting, action, notable details, mood>",
 "setting": {"indoor_outdoor": "indoor|outdoor|mixed|unknown", "place_type": "<e.g. kitchen, pub, theatre, beach, city street, forest, museum>", "place_guess": "<country/city/landmark only if clearly evidenced by signs, landmarks or GPS, else 'unknown'>", "time_of_day": "dawn|day|golden hour|dusk|night|unknown", "weather": "sunny|overcast|rain|snow|fog|n/a|unknown", "season": "spring|summer|autumn|winter|unknown"},
 "event": "<e.g. birthday party, wedding, hike, holiday, concert, everyday, unknown>",
 "people": {"count": <int>, "children": <int>, "main_subject_is_person": true|false, "faces": [{"box": [x, y, w, h], "size": "small|medium|large", "facing": "camera|away|profile"}]},
 "objects": [{"label": "<noun>", "box": [x, y, w, h]}],
 "text_visible": ["<legible text, signs, labels>"],
 "tags": ["<10-25 lowercase keywords useful for search: prefer plain common nouns and these where they apply: portrait, group, child, baby, family, friends, couple, selfie, landscape, cityscape, street, architecture, interior, food, drink, animal, pet, dog, cat, bird, plant, flower, tree, vehicle, car, boat, train, plane, bicycle, sport, hiking, climbing, swimming, beach, sea, lake, river, mountain, forest, park, garden, snow, sunset, night, concert, theatre, museum, restaurant, pub, party, birthday, wedding, christmas, holiday, travel, home, kitchen, living room, bedroom, office, school, playground, festival, market, church, castle, bridge, sign, document, screen, art, close-up, macro, silhouette, reflection>"],
 "lighting": {"sources": [{"type": "daylight|overcast sky|shade|sunset sun|tungsten|halogen|fluorescent|LED white|LED coloured|stage lights|candle|fire|flash|screen|street light|mixed", "colour": "<warm|neutral|cool|colour name>", "share": <0-1 of the picture lit by it>}], "mixed": true|false, "direction": "front|side|back|top|diffuse|unknown", "contrast": "low|medium|high", "level": "very dark|dark|normal|bright|very bright"},
 "colour": {"cast_now": "<cast you see in this rendering, e.g. 'strong orange', 'slightly green', 'none'>", "neutral_refs": [{"what": "<surface that should be white/grey in reality>", "box": [x, y, w, h], "looks": "<how it looks now>"}], "skin_visible": true|false, "skin_looks": "<natural|too orange|too red|too pale|too green|n/a>", "dominant": ["<2-4 colour names>"]},
 "exposure": {"intended_dark": true|false, "clipped_highlights": "none|some|large", "crushed_shadows": "none|some|large", "noise": "low|medium|high", "sharpness": "sharp|soft|motion blur|out of focus"},
 "composition": {"main_subject_box": [x, y, w, h], "horizon_tilt_deg": <number or null>, "orientation_ok": true|false},
 "quality": {"technical": <1-5>, "aesthetic": <1-5>, "keeper": true|false},
 "sensitive": {"nudity": "none|partial|explicit", "child_focus": true|false, "private_document": true|false},
 "confidence": {"lighting": <1-3>, "place": <1-3>}
}
Boxes are [x, y, width, height] as fractions 0-1 of the image (x, y = top-left). At most 12 objects, the most important first.
