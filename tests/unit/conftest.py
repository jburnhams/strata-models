"""Unit-suite only: the hypothesis profile (isolated home and no network are autouse in tests/conftest.py for every suite)."""
from hypothesis import HealthCheck, settings

settings.register_profile('unit', deadline=None, max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
settings.load_profile('unit')
