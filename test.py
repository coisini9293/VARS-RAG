#!/usr/bin/env python3
"""Entry point: batch RAG / prompt / CoT / self-consistency experiments (see experiment/)."""
from experiment.ensure_local_models import ensure_experiment_models
from experiment.runner import main

if __name__ == "__main__":
    ensure_experiment_models()
    main()
