"""Which code decides an experiment's analysis, as a hash: a kept analysis (evals.summaries) is redone when this changes and only then.

Not a hash of whole files: a logging line or a comment in summaries.py made every kept analysis out of date (2026-11-25: 26 CPU minutes after a
restart), and an edit to one metric redid every experiment that does not use it, while ink.py and similarity.py were not looked at at all. Here
the functions reachable from `core.analyse` and from the experiment's own metrics are followed through this package's imports and their syntax
trees hashed, docstrings and comments left out. Static: a name is followed wherever it appears, so it errs towards redoing."""

import ast
import functools
import hashlib
import os

from stratamodels.evals import metrics as M

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = ("evals.core", "analyse")
METRICS = ("evals.metrics", "METRICS")  # the table of them all: only an experiment's own entries count
# what an analysis calls but is not decided by: the database, settings, the queue, the cloud CLIs (judges' verdicts are rows it reads)
SKIP = {"evals.cloud", "evals.api", "evals.report", "evals.summaries", "evals.codehash"}


def _followed(module):
    return module not in SKIP and (
        module.startswith(("evals.", "workers.")) or module in ("runtimes.pageclean", "runtimes.fontfit")
    )


class _Module:
    """A source file's top-level definitions (name -> nodes) and what its imports, anywhere in it, call things."""

    def __init__(self, name):
        with open(os.path.join(PACKAGE, *name.split(".")) + ".py", encoding="utf8") as f:
            tree = ast.parse(f.read())
        self.defs, self.modules, self.names = {}, {}, {}
        for n in tree.body:
            for t in _targets(n):
                self.defs.setdefault(t, []).append(n)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] == "stratamodels":
                base = n.module.partition(".")[2]
                for a in n.names:
                    sub = f"{base}.{a.name}" if base else a.name
                    if os.path.isfile(os.path.join(PACKAGE, *sub.split(".")) + ".py"):
                        self.modules[a.asname or a.name] = sub
                    else:
                        self.names[a.asname or a.name] = (base, a.name)


def _targets(n):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [n.name]
    if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        ts = n.targets if isinstance(n, ast.Assign) else [n.target]
        return [x.id for t in ts for x in ast.walk(t) if isinstance(x, ast.Name)]
    return []


@functools.cache
def _module(name):
    return _Module(name)


def _plain(node):
    """The node's syntax, docstrings left out."""
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            n.body = body[1:] or [ast.Pass()]
    return ast.dump(node)


@functools.cache
def _definition(module, name):
    """(hash of one top-level definition, the definitions it names)."""
    m = _module(module)
    refs, text = set(), []
    for node in m.defs[name]:
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in m.modules:
                refs.add((m.modules[n.value.id], n.attr))
            elif isinstance(n, ast.Name):
                if n.id in m.names:
                    refs.add(m.names[n.id])
                elif n.id in m.defs:
                    refs.add((module, n.id))
        text.append(_plain(node))
    return hashlib.sha256("\n".join(text).encode()).hexdigest(), frozenset(refs)


def _reach(roots, judges=True):
    """{(module, name): hash} of every definition reachable from `roots` (analyse calls evals.judges only for an experiment with judges)."""
    out, todo = {}, list(roots)
    while todo:
        ref = todo.pop()
        if not judges and ref[0] == "evals.judges":
            continue
        if ref in out or ref == METRICS or not _followed(ref[0]) or ref[1] not in _module(ref[0]).defs:
            continue
        h, refs = _definition(*ref)
        out[ref] = h
        todo += refs
    return out


def _metric_roots(name):
    """(the definitions behind one metric, what else decides it): its functions, and the values a made one (`_reader(group, key)`) closed over."""
    m, roots, extra = M.METRICS[name], set(), []
    for f in (m.score, m.dataset, m.prepare, m.on):
        if f is None:
            continue
        top = f.__qualname__.split(".")[0]
        if top == "<lambda>":  # written inside the table: the whole table stands for it
            return roots, None
        roots.add((f.__module__.partition(".")[2], top))
        extra += [repr(c.cell_contents) for c in (f.__closure__ or ()) if not callable(c.cell_contents)]
        roots |= {
            (c.cell_contents.__module__.partition(".")[2], c.cell_contents.__qualname__.split(".")[0])
            for c in (f.__closure__ or ())
            if hasattr(c.cell_contents, "__qualname__")
        }
    return roots, extra


@functools.lru_cache(maxsize=512)
def _code(metrics, judges):
    roots, parts = {ROOT}, []
    for n in metrics:
        if n not in M.METRICS:
            continue
        r, extra = _metric_roots(n)
        if extra is None:
            parts.append(_definition(*METRICS)[0])
            r = set(_definition(*METRICS)[1])
        roots |= r
        parts.append(repr((M.METRICS[n].to_json(), extra)))
    parts += [f"{m}.{n} {h}" for (m, n), h in sorted(_reach(roots, judges).items())]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def code(spec):
    """The hash of the code that decides `spec`'s analysis."""
    return _code(tuple(sorted({*spec.metrics, spec.primary, *spec.guards})), bool(spec.judges))
