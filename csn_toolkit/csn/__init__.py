"""
Clifford Space Network (CSN) Core Toolkit.
Geometric Deep Learning via Clifford Lie Rotors and Associative Scans.
"""

from csn.scan import (
    CliffordScanFunction,
    parallel_rotor_scan,
    rotor_binary_op,
    surrogate_spike
)

from csn.rotors import (
    cayley_rotor_4x4,
    cayley_rotor_8x8,
    cayley_rotor_16x16,
    schulz_inverse_4x4,
    schulz_inverse_16x16
)

from csn.loss import (
    compute_sasaki_ciou_vectorized,
    draw_gaussian
)

from csn.vision import (
    SpatialPatchStem,
    CliffordSpatialRotorBlockCl4,
    CliffordSpatialRotorBlockCl8,
    CliffordFPN,
    CliffordDetectionHead,
    CliffordDetModel,
    ModelEMA
)

from csn.transformer import (
    CliffordCausalRotorBlock,
    CliffordTransformerBlock,
    CliffordTransformerLM
)

__version__ = "1.0.0"

__all__ = [
    "CliffordScanFunction",
    "parallel_rotor_scan",
    "rotor_binary_op",
    "surrogate_spike",
    "cayley_rotor_4x4",
    "cayley_rotor_8x8",
    "cayley_rotor_16x16",
    "schulz_inverse_4x4",
    "schulz_inverse_16x16",
    "compute_sasaki_ciou_vectorized",
    "draw_gaussian",
    "SpatialPatchStem",
    "CliffordSpatialRotorBlockCl4",
    "CliffordSpatialRotorBlockCl8",
    "CliffordFPN",
    "CliffordDetectionHead",
    "CliffordDetModel",
    "ModelEMA",
    "CliffordCausalRotorBlock",
    "CliffordTransformerBlock",
    "CliffordTransformerLM"
]
