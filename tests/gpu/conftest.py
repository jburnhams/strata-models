"""The GPU suite may download weights on first run (Hugging Face, insightface): the network guard of the other suites is switched off here."""

import pytest


@pytest.fixture(autouse=True)
def no_network():
    pass
