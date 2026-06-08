"""Convenience shim: ``python predict.py <image>`` == the ``hivevision`` console script."""

from hivevision.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
