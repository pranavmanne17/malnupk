"""MalnuPK - pharmacokinetics for undernourished, toxicant-exposed patients.

A research simulator and deep-learning model. NOT a clinical tool.
"""
__version__ = "0.1.0"

from .physiology import PopulationSampler, VirtualPatient            # noqa: F401
from .toxicants import ToxicantSampler, TOXICANTS, EXPOSURE_SETTINGS  # noqa: F401
from .drugs import DRUGS, Drug                                        # noqa: F401
from .pk import (DosingRegimen, derive_parameters, simulate_patient,  # noqa: F401
                 exposure_metrics)

DISCLAIMER = (
    "MalnuPK is a mechanistic simulator and a model trained on its output. "
    "It has not been validated against clinical data and must not be used to "
    "make dosing decisions for any real patient."
)
