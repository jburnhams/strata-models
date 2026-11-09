"""The registry: which model serves each role, and every model the service knows. The service reads it from its tables (models, roles,
model_roles, preferences: `Registry.from_db`); `models.toml` (in this package, with the data folder's `models.toml` merged over it table by table)
is the SEED: `seed()` adds what is new and updates rows from the file while nobody has edited them here (from the web app, the API or the CLI),
so a model added to the file in a pull request still arrives, and a choice made here is not undone by the next start.
A model whose runtime is not one this service implements is still listed ("known, not served here yet"): the Models page shows which apps
still load it themselves, which is the to-do list for moving them over."""

import dataclasses, hashlib, json, os, time, tomllib

from stratamodels import runtimes, settings

BUILTIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.toml")


class RegistryError(ValueError):
    """The registry file is wrong, or a job asked for a role or model that does not exist."""


@dataclasses.dataclass(frozen=True)
class Model:
    key: str
    runtime: str
    name: str
    roles: tuple
    vram_gb: float
    licence: str = ""
    source: str = ""
    used_by: tuple = ()
    options: dict = dataclasses.field(default_factory=dict)

    @property
    def served(self):
        return self.runtime in runtimes.RUNTIMES

    def to_json(self):
        return dict(
            key=self.key,
            runtime=self.runtime,
            name=self.name,
            roles=list(self.roles),
            vram_gb=self.vram_gb,
            licence=self.licence,
            source=self.source,
            used_by=list(self.used_by),
            options=self.options,
            served=self.served,
        )


@dataclasses.dataclass(frozen=True)
class Role:
    key: str
    description: str
    model: str

    def to_json(self):
        return dict(key=self.key, description=self.description, model=self.model)


def _merge(base, extra):
    out = {k: dict(v) for k, v in base.items()}
    for k, v in extra.items():
        out[k] = {**out.get(k, {}), **v}
    return out


def _read(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise RegistryError(f"{path}: {e}")


@dataclasses.dataclass(frozen=True)
class Preference:
    app: str
    role: str
    model: str | None = None
    params: dict = dataclasses.field(default_factory=dict)
    note: str = ""

    def to_json(self):
        return dataclasses.asdict(self)


class Registry:
    def __init__(self, roles, models, preferences=None):
        self.roles, self.models, self.preferences = roles, models, dict(preferences or {})
        for r in roles.values():
            if r.model not in models:
                raise RegistryError(f"role {r.key!r} names model {r.model!r}, which is not in the registry")
            if r.key not in models[r.model].roles:
                raise RegistryError(
                    f"role {r.key!r} names model {r.model!r}, whose roles are {list(models[r.model].roles)}"
                )

    @classmethod
    def from_dict(cls, d):
        models = {}
        for k, m in (d.get("models") or {}).items():
            try:
                models[k] = Model(
                    key=k,
                    runtime=str(m["runtime"]),
                    name=str(m.get("name", k)),
                    roles=tuple(m.get("roles") or ()),
                    vram_gb=float(m.get("vram_gb", 1.0)),
                    licence=str(m.get("licence", "")),
                    source=str(m.get("source", "")),
                    used_by=tuple(m.get("used_by") or ()),
                    options=dict(m.get("options") or {}),
                )
            except (KeyError, TypeError, ValueError) as e:
                raise RegistryError(f"model {k!r}: {e!r}")
        roles = {}
        for k, r in (d.get("roles") or {}).items():
            if "model" not in r:
                raise RegistryError(f"role {k!r} names no model")
            roles[k] = Role(key=k, description=str(r.get("description", "")), model=str(r["model"]))
        return cls(roles, models)

    @classmethod
    def load(cls, builtin=BUILTIN, override=None):
        """The built-in registry with the data folder's `models.toml` merged over it."""
        base = _read(builtin)
        extra = _read(override or os.path.join(settings.data_dir(), "models.toml"))
        return cls.from_dict(
            dict(
                roles=_merge(base.get("roles") or {}, extra.get("roles") or {}),
                models=_merge(base.get("models") or {}, extra.get("models") or {}),
            )
        )

    def preference(self, app, role):
        return self.preferences.get((app, role))

    def resolve(self, role, model=None, app=None):
        """The model a job for `role` runs on: `model` if it is listed under that role, else the app's preferred one, else the role's own. Raises
        RegistryError otherwise."""
        if role not in self.roles:
            raise RegistryError(f"no role {role!r} (known: {', '.join(sorted(self.roles))})")
        pref = self.preference(app, role) if app else None
        key = model or (pref and pref.model) or self.roles[role].model
        m = self.models.get(key) or self.models.get(key.split("~")[0])
        if m is not None and m.key != key and "~" in key:
            m = None  # a variant that is not registered (yet) is not its base model
        if m is None or role not in m.roles:
            raise RegistryError(f"model {key!r} does not serve role {role!r}")
        if not m.served:
            raise RegistryError(
                f"model {key!r} ({m.runtime}) is known but not served here yet; the apps listed load it themselves"
            )
        return m

    def variant(self, key, options):
        """`key` with some load options changed (an eval candidate: det_size 1024, int8): a model of its own, `key~<hash>`, loaded and freed apart
        from the original. Registered on first use; listed on the Models page while it exists."""
        import hashlib, json

        base = self.models.get(key)
        if base is None:
            raise RegistryError(f"no model {key!r}")
        if not options:
            return base
        merged = {**base.options, **options}
        vkey = f"{base.key}~{hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()[:8]}"
        if vkey not in self.models:
            self.models[vkey] = dataclasses.replace(base, key=vkey, options=merged, used_by=("evals",))
        return self.models[vkey]

    def to_json(self):
        return dict(
            roles=[r.to_json() for r in self.roles.values()],
            models=[m.to_json() for m in self.models.values()],
            preferences=[p.to_json() for p in self.preferences.values()],
        )

    @classmethod
    def from_db(cls, engine, keep=None):
        """The registry in the database; `keep` (the registry this replaces) hands over its variants (`key~hash`, built at run time, never stored)."""
        from sqlalchemy import select
        from stratamodels.db import session
        from stratamodels.db.models import ModelRole, ModelRow, Preference as P, RoleRow

        with session(engine) as s:
            serves = {}
            for mr in s.scalars(select(ModelRole).order_by(ModelRole.model, ModelRole.position, ModelRole.role)):
                serves.setdefault(mr.model, []).append(mr.role)
            models = {
                r.key: Model(
                    key=r.key,
                    runtime=r.runtime,
                    name=r.name,
                    roles=tuple(serves.get(r.key, ())),
                    vram_gb=float(r.vram_gb),
                    licence=r.licence,
                    source=r.source,
                    used_by=tuple(r.used_by or ()),
                    options=dict(r.options or {}),
                )
                for r in s.scalars(select(ModelRow).order_by(ModelRow.key))
            }
            roles = {
                r.key: Role(key=r.key, description=r.description, model=r.model)
                for r in s.scalars(select(RoleRow).order_by(RoleRow.key))
            }
            prefs = {
                (p.app, p.role): Preference(
                    app=p.app, role=p.role, model=p.model, params=dict(p.params or {}), note=p.note or ""
                )
                for p in s.scalars(select(P).order_by(P.app, P.role))
            }
        if keep is not None:
            models.update({k: v for k, v in keep.models.items() if "~" in k and k.split("~")[0] in models})
        return cls(roles, models, prefs)


# ---- the tables -------------------------------------------------------------------------------------------------------------------------------
def _hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _model_fields(k, m):
    m = Model(
        key=k,
        runtime=str(m["runtime"]),
        name=str(m.get("name", k)),
        roles=tuple(m.get("roles") or ()),
        vram_gb=float(m.get("vram_gb", 1.0)),
        licence=str(m.get("licence", "")),
        source=str(m.get("source", "")),
        used_by=tuple(m.get("used_by") or ()),
        options=dict(m.get("options") or {}),
    )
    return dict(
        runtime=m.runtime,
        name=m.name,
        vram_gb=m.vram_gb,
        licence=m.licence,
        source=m.source,
        used_by=list(m.used_by),
        options=m.options,
    ), list(m.roles)


def _set_roles(s, key, roles):
    from sqlalchemy import delete
    from stratamodels.db.models import ModelRole

    s.execute(delete(ModelRole).where(ModelRole.model == key))
    for i, r in enumerate(dict.fromkeys(roles)):
        s.add(ModelRole(model=key, role=str(r), position=i))


def file_dict(builtin=BUILTIN, override=None):
    """models.toml with the data folder's merged over it, as one dict (roles, models)."""
    base = _read(builtin)
    extra = _read(override or os.path.join(settings.data_dir(), "models.toml"))
    return dict(
        roles=_merge(base.get("roles") or {}, extra.get("roles") or {}),
        models=_merge(base.get("models") or {}, extra.get("models") or {}),
    )


def seed(engine, builtin=BUILTIN, override=None, clock=time.time, log=None):
    """Bring models.toml into the tables: entries not in the database are added; an entry whose text changed since it was last imported updates
    its row, unless the row was edited here (then the database's choice stands, and the skip is reported). Returns {added, updated, kept}."""
    from stratamodels.db import session
    from stratamodels.db.models import ModelRow, RoleRow

    d = file_dict(builtin, override)
    Registry.from_dict(d)  # a broken file is refused before anything is written
    out = dict(added=[], updated=[], kept=[])
    now = clock()
    with session(engine) as s:
        for k, m in d["models"].items():
            h = _hash(m)
            fields, roles = _model_fields(k, m)
            row = s.get(ModelRow, k)
            if row is None:
                s.add(ModelRow(key=k, updated_at=now, seed_hash=h, **fields))
                s.flush()
                _set_roles(s, k, roles)
                out["added"].append(f"model {k}")
            elif row.seed_hash != h:
                if row.edited:
                    out["kept"].append(f"model {k}")
                    continue
                for f, v in fields.items():
                    setattr(row, f, v)
                row.seed_hash, row.updated_at = h, now
                _set_roles(s, k, roles)
                out["updated"].append(f"model {k}")
        s.flush()
        for k, r in d["roles"].items():
            h = _hash(r)
            row = s.get(RoleRow, k)
            if row is None:
                s.add(
                    RoleRow(
                        key=k,
                        model=str(r["model"]),
                        description=str(r.get("description", "")),
                        updated_at=now,
                        seed_hash=h,
                    )
                )
                out["added"].append(f"role {k}")
            elif row.seed_hash != h:
                if row.edited:
                    out["kept"].append(f"role {k}")
                    continue
                row.model, row.description, row.seed_hash, row.updated_at = (
                    str(r["model"]),
                    str(r.get("description", "")),
                    h,
                    now,
                )
                out["updated"].append(f"role {k}")
        s.flush()
        try:
            _check(s)  # file and database each valid can still merge into a broken registry
        except RegistryError as e:
            s.rollback()
            raise RegistryError(
                f"models.toml not imported (nothing written): with the rows edited here it would leave the registry inconsistent: {e}"
            )
        s.commit()
    if log and (out["added"] or out["updated"] or out["kept"]):
        log(
            f"registry seed: added {len(out['added'])}, updated {len(out['updated'])}, kept as edited here {out['kept'] or 'none'}"
        )
    return out


def _edit(engine, fn):
    """Run an edit and refuse it (nothing written) if it leaves a role on a model that is missing or does not serve it."""
    from stratamodels.db import session
    from sqlalchemy.exc import IntegrityError

    with session(engine) as s:
        try:
            fn(s)
            s.flush()
        except IntegrityError as e:
            s.rollback()
            raise RegistryError(
                f"refused: a role or preference would name a model that is not in the registry ({e.orig})"
            )
        try:
            _check(s)
        except RegistryError:
            s.rollback()
            raise
        s.commit()
    return Registry.from_db(engine)


def _check(s):
    """Raise RegistryError if the rows in this session leave a role on a model that is missing or does not serve it."""
    from sqlalchemy import select
    from stratamodels.db.models import ModelRole, ModelRow, RoleRow

    models = {r.key for r in s.scalars(select(ModelRow))}
    serves = {(m.model, m.role) for m in s.scalars(select(ModelRole))}
    for r in s.scalars(select(RoleRow)):
        if r.model not in models:
            raise RegistryError(f"role {r.key!r} names model {r.model!r}, which is not in the registry")
        if (r.model, r.key) not in serves:
            raise RegistryError(f"role {r.key!r} names model {r.model!r}, which does not list that role")


def set_role(engine, key, model=None, description=None, clock=time.time):
    """Choose the model for a role (or create the role); the row stops following models.toml."""
    from stratamodels.db.models import RoleRow

    def fn(s):
        row = s.get(RoleRow, key)
        if row is None:
            if not model:
                raise RegistryError(f"a new role {key!r} needs a model")
            s.add(RoleRow(key=key, model=model, description=description or "", updated_at=clock(), edited=True))
            return
        if model is not None:
            row.model = model
        if description is not None:
            row.description = description
        row.edited, row.updated_at = True, clock()

    return _edit(engine, fn)


MODEL_FIELDS = ("runtime", "name", "vram_gb", "licence", "source", "used_by", "options")


def save_model(engine, key, roles=None, clock=time.time, **fields):
    """Add a model or change some of its fields (and, with `roles`, the roles it may serve); the row stops following models.toml."""
    from stratamodels.db.models import ModelRow

    bad = set(fields) - set(MODEL_FIELDS)
    if bad:
        raise RegistryError(f"unknown model fields {sorted(bad)}")
    if "~" in key:
        raise RegistryError("a key with ~ is a variant, built at run time")

    def fn(s):
        row = s.get(ModelRow, key)
        if row is None:
            if "runtime" not in fields:
                raise RegistryError(f"a new model {key!r} needs a runtime")
            f, _ = _model_fields(key, dict(fields))
            s.add(ModelRow(key=key, updated_at=clock(), edited=True, **f))
            s.flush()
        else:
            cur = {f: getattr(row, f) for f in MODEL_FIELDS}
            f, _ = _model_fields(key, {**cur, **fields})
            for k, v in f.items():
                setattr(row, k, v)
            row.edited, row.updated_at = True, clock()
        if roles is not None:
            _set_roles(s, key, list(roles))

    try:
        return _edit(engine, fn)
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, RegistryError):
            raise
        raise RegistryError(f"model {key!r}: {e!r}")


def follow_seed(engine, kind, key):
    """Let a model or role follow models.toml again (its edits are replaced at the next seed)."""
    from stratamodels.db import session
    from stratamodels.db.models import ModelRow, RoleRow

    with session(engine) as s:
        row = s.get(ModelRow if kind == "model" else RoleRow, key)
        if row is None:
            raise RegistryError(f"no {kind} {key!r}")
        row.edited, row.seed_hash = False, None
        s.commit()


def set_preference(engine, app, role, model=None, params=None, note=None, clock=time.time):
    """An app's own model and default params for a role (both None: remove the preference)."""
    from sqlalchemy import select
    from stratamodels.db import session
    from stratamodels.db.models import Preference as P, RoleRow

    reg = Registry.from_db(engine)
    if role not in reg.roles:
        raise RegistryError(f"no role {role!r}")
    if model is not None:
        reg.resolve(role, model)
    if params is not None and not isinstance(params, dict):
        raise RegistryError("params is an object")
    with session(engine) as s:
        row = s.scalar(select(P).where(P.app == app, P.role == role))
        if model is None and not params:
            if row is not None:
                s.delete(row)
        elif row is None:
            s.add(P(app=app, role=role, model=model, params=params or None, note=note, updated_at=clock()))
        else:
            row.model, row.params, row.note, row.updated_at = model, params or None, note, clock()
        s.commit()
    return Registry.from_db(engine)
