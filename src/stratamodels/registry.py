"""The registry: which model serves each role, read from `models.toml` (in this package), optionally overridden by `models.toml` in the data folder.

The override file is merged table by table, so a machine can change one role's model or one model's options without copying the whole file.
A model whose runtime is not one this service implements is still listed ("known, not served here yet"): the Models page shows which apps
still load it themselves, which is the to-do list for moving them over."""
import dataclasses, os, tomllib

from stratamodels import runtimes, settings

BUILTIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.toml')


class RegistryError(ValueError):
    """The registry file is wrong, or a job asked for a role or model that does not exist."""


@dataclasses.dataclass(frozen=True)
class Model:
    key: str
    runtime: str
    name: str
    roles: tuple
    vram_gb: float
    licence: str = ''
    source: str = ''
    used_by: tuple = ()
    options: dict = dataclasses.field(default_factory=dict)

    @property
    def served(self): return self.runtime in runtimes.RUNTIMES

    def to_json(self): return dict(key=self.key, runtime=self.runtime, name=self.name, roles=list(self.roles), vram_gb=self.vram_gb, licence=self.licence, source=self.source, used_by=list(self.used_by), options=self.options, served=self.served)


@dataclasses.dataclass(frozen=True)
class Role:
    key: str
    description: str
    model: str

    def to_json(self): return dict(key=self.key, description=self.description, model=self.model)


def _merge(base, extra):
    out = {k: dict(v) for k, v in base.items()}
    for k, v in extra.items(): out[k] = {**out.get(k, {}), **v}
    return out


def _read(path):
    try:
        with open(path, 'rb') as f: return tomllib.load(f)
    except FileNotFoundError: return {}
    except tomllib.TOMLDecodeError as e: raise RegistryError(f'{path}: {e}')


class Registry:
    def __init__(self, roles, models):
        self.roles, self.models = roles, models
        for r in roles.values():
            if r.model not in models: raise RegistryError(f'role {r.key!r} names model {r.model!r}, which is not in the registry')
            if r.key not in models[r.model].roles: raise RegistryError(f'role {r.key!r} names model {r.model!r}, whose roles are {list(models[r.model].roles)}')

    @classmethod
    def from_dict(cls, d):
        models = {}
        for k, m in (d.get('models') or {}).items():
            try: models[k] = Model(key=k, runtime=str(m['runtime']), name=str(m.get('name', k)), roles=tuple(m.get('roles') or ()), vram_gb=float(m.get('vram_gb', 1.0)), licence=str(m.get('licence', '')), source=str(m.get('source', '')), used_by=tuple(m.get('used_by') or ()), options=dict(m.get('options') or {}))
            except (KeyError, TypeError, ValueError) as e: raise RegistryError(f'model {k!r}: {e!r}')
        roles = {}
        for k, r in (d.get('roles') or {}).items():
            if 'model' not in r: raise RegistryError(f'role {k!r} names no model')
            roles[k] = Role(key=k, description=str(r.get('description', '')), model=str(r['model']))
        return cls(roles, models)

    @classmethod
    def load(cls, builtin=BUILTIN, override=None):
        """The built-in registry with the data folder's `models.toml` merged over it."""
        base = _read(builtin); extra = _read(override or os.path.join(settings.data_dir(), 'models.toml'))
        return cls.from_dict(dict(roles=_merge(base.get('roles') or {}, extra.get('roles') or {}), models=_merge(base.get('models') or {}, extra.get('models') or {})))

    def resolve(self, role, model=None):
        """The model a job for `role` runs on: the role's own, or `model` if it is listed under that role. Raises RegistryError otherwise."""
        if role not in self.roles: raise RegistryError(f'no role {role!r} (known: {", ".join(sorted(self.roles))})')
        key = model or self.roles[role].model
        m = self.models.get(key)
        if m is None or role not in m.roles: raise RegistryError(f'model {key!r} does not serve role {role!r}')
        if not m.served: raise RegistryError(f'model {key!r} ({m.runtime}) is known but not served here yet; the apps listed load it themselves')
        return m

    def to_json(self): return dict(roles=[r.to_json() for r in self.roles.values()], models=[m.to_json() for m in self.models.values()])
