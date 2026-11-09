"""strata-models: one service on the machine that owns the GPU. It loads the chosen model for each role, runs jobs from every app through one queue, lends the GPU to apps that run their own kernels, and shows all of it in a web app."""

__version__ = "0.1.0"
