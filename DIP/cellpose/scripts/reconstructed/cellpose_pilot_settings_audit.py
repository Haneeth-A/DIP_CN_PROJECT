"""
Cellpose pilot settings audit (reconstructed helper).

This script reports the installed Cellpose version and the callable signatures
available in the current environment. It does NOT prove which settings were
used in a past Colab session.
"""
import inspect

try:
    import cellpose
    from cellpose import models
    print("Cellpose version:", getattr(cellpose, "__version__", "unknown"))
    print("CellposeModel signature:")
    print(inspect.signature(models.CellposeModel))
    print("CellposeModel.eval signature:")
    print(inspect.signature(models.CellposeModel.eval))
except Exception as exc:
    print("Could not inspect Cellpose in this environment:", repr(exc))
    print("Install the same Cellpose version used by the project, if known.")
