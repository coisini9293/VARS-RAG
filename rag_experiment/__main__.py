"""Allow: python -m rag_experiment"""
from __future__ import annotations

import sys


def _entry() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        from rag_experiment.runner import main

        main()
        return

    from experiment.ensure_local_models import ensure_experiment_models
    from rag_experiment.runner import main

    ensure_experiment_models()
    main()


if __name__ == "__main__":
    _entry()
