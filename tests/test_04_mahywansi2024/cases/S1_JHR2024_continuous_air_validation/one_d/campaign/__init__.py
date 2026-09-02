"""Result-independent Stage-1 -> Stage-2 campaign protocol for S1.

This package deliberately owns orchestration only.  It does not implement or
promote the horizontal, riser, T-node, plume, or trajectory-observer physics.
"""

from .config import (
    DEFAULT_CONFIG_PATH,
    CampaignProtocolConfig,
    load_campaign_protocol_config,
)
from .contracts import (
    AcceptedCommonState,
    BoundaryCommand,
    ExactAdvanceRunner,
    ObservationBridge,
    Stage1AcceptanceCandidate,
    Stage1BoundaryFlows,
    Stage1Observation,
    StateCodec,
)
from .orchestrator import (
    CampaignProtocolError,
    FormalCampaignResult,
    PreproductionSmokeResult,
    run_formal_campaign,
    run_preproduction_smoke,
)
from .stability import (
    Stage1StabilityReport,
    evaluate_stage1_stability,
)

__all__ = [
    "AcceptedCommonState",
    "BoundaryCommand",
    "CampaignProtocolConfig",
    "CampaignProtocolError",
    "DEFAULT_CONFIG_PATH",
    "ExactAdvanceRunner",
    "FormalCampaignResult",
    "ObservationBridge",
    "PreproductionSmokeResult",
    "Stage1AcceptanceCandidate",
    "Stage1BoundaryFlows",
    "Stage1Observation",
    "Stage1StabilityReport",
    "StateCodec",
    "evaluate_stage1_stability",
    "load_campaign_protocol_config",
    "run_formal_campaign",
    "run_preproduction_smoke",
]
