"""Independent MMFF94 tests of localized and synchronized propeller motion."""
import os

# The force-field context is serial. Small dense Hessians are faster and more
# reproducible without nested BLAS oversubscription. Set before importing numpy.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = os.environ.get("PROPELLER_BLAS_THREADS", "1")

__version__ = "0.1.0"
