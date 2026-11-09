"""Build the eval dataset `image-prompts-20`: twenty prompts that cover what local image models get wrong (hands, text in the picture, counting,
several subjects, photographs against illustrations, small faces, reflections), the same for every image experiment. Run once:

    .venv/bin/python scripts/evals/image_prompts.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from stratamodels.evals import core as C

PROMPTS = [
    ('portrait', 'A close-up portrait photo of an elderly fisherman with a weathered face, overcast light, shallow depth of field'),
    ('hands', 'Two hands kneading bread dough on a floured wooden table, seen from above, natural window light'),
    ('text', 'A shop sign above a bakery door that reads "FRESH BREAD DAILY" in painted gold letters'),
    ('counting', 'Exactly five red apples in a row on a white shelf, studio photo'),
    ('scene', 'A trail runner on a muddy forest path at dawn, mist between the trees, action photo'),
    ('landscape', 'A wide mountain valley with a glacial lake, late afternoon sun, photograph'),
    ('animals', 'A border collie herding sheep on a green hillside, telephoto photograph'),
    ('child-art', 'A crayon drawing of a house, a sun and three stick figures, as drawn by a five-year-old'),
    ('architecture', 'The interior of a gothic cathedral, light through stained glass, wide angle'),
    ('food', 'A bowl of ramen with a soft-boiled egg, steam rising, top-down food photography'),
    ('night', 'A city street at night in the rain, neon reflections on wet tarmac'),
    ('product', 'A pair of running shoes on a plain grey background, studio product photo'),
    ('several', 'A woman in a yellow raincoat and a man in a blue jacket sharing an umbrella at a bus stop'),
    ('small-faces', 'A crowd of runners at the start line of a marathon, seen from a bridge above'),
    ('reflection', 'A cat looking at its reflection in a mirror, the reflection matching exactly'),
    ('style-watercolour', 'A watercolour painting of a lighthouse on a cliff, loose brush strokes'),
    ('diagram', 'A clean flat illustration of the water cycle with arrows, no text'),
    ('macro', 'A macro photo of a dewdrop on a spider web, morning light, bokeh'),
    ('vehicle', 'A red vintage bicycle leaning against a brick wall, photograph'),
    ('indoor', 'A cosy living room with a fireplace, an armchair and a sleeping dog, warm evening light'),
]


def main():
    items = [dict(id=k, input=dict(prompt=p, seed=1000 + i), tags=dict(kind=k)) for i, (k, p) in enumerate(PROMPTS)]
    ds = C.write_dataset('image-prompts-20', items, description='Twenty prompts covering hands, text, counting, several subjects, small faces, reflections and styles; one fixed seed each')
    print(f'{len(ds.items)} prompts, version {ds.version}')


if __name__ == '__main__':
    main()
