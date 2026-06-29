"""karyon — a legible reliability/QC + DBTL layer over commodity bio-AI tools.

The public surface is the qualify spine: `karyon.qualify(artifact, modality)` runs the right deterministic
gate and returns a `QualifyResult` whose `.to_dict()` is a stable JSON schema. The same thing is available
on the command line as `karyon qualify ...`. The underlying contract engine (`Verdict`, `Reason`,
`Contract`, `ContractSet`) is re-exported for callers that build their own gates.
"""

from .contracts import Contract, ContractSet, Reason, Verdict
from .repair import (Agent, DnaRepairAgent, DnaSpec, MolRepairAgent, MolSpec,
                     RepairStep, RepairTrajectory, format_trajectory, repair_loop)
from .spine import GATES, Gate, QualifyError, QualifyResult, modalities, qualify

__version__ = "0.2.0"

__all__ = [
    "qualify",
    "QualifyResult",
    "QualifyError",
    "Gate",
    "GATES",
    "modalities",
    "Verdict",
    "Reason",
    "Contract",
    "ContractSet",
    # the agent self-repair loop
    "repair_loop",
    "RepairTrajectory",
    "RepairStep",
    "Agent",
    "DnaRepairAgent",
    "DnaSpec",
    "MolRepairAgent",
    "MolSpec",
    "format_trajectory",
    "__version__",
]
