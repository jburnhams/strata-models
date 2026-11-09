import contextlib

from sqlalchemy.orm import Session


@contextlib.contextmanager
def session(engine, **kw):
    """A session on the service's engine, under the store's lock when there is one (the in-memory SQLite store shares one connection)."""
    with getattr(engine, 'stratamodels_lock', None) or contextlib.nullcontext(), Session(engine, **kw) as s: yield s
