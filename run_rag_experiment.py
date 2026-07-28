#!/usr/bin/env python3
"""Entry point for dedicated RAG ablation experiments (see rag_experiment/)."""
from __future__ import annotations

import sys


def main() -> None:
    # Avoid heavy downloads when user only asks for help.
    if "-h" in sys.argv or "--help" in sys.argv:
        from rag_experiment.runner import main as run_main

        run_main()
        return

    from experiment.ensure_local_models import ensure_experiment_models
    from rag_experiment.runner import main as run_main

    ensure_experiment_models()
    run_main()


if __name__ == "__main__":
    main()
