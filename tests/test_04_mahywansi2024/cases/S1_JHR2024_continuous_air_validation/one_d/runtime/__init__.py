"""Concrete, fail-closed runtime adapters for the S1 campaign protocol."""

from .campaign_adapter import (
    DEFAULT_SOURCE_CONTRACT_PATH,
    PREPRODUCTION_SMOKE_DURATION_S,
    S1CampaignExactAdvanceAdapter,
    S1CampaignRuntimeBundle,
    S1CampaignRuntimeError,
    S1CoupledStateCodec,
    S1Stage1ObservationBridge,
    build_current_s1_campaign_runtime,
)

__all__ = [
    "DEFAULT_SOURCE_CONTRACT_PATH",
    "PREPRODUCTION_SMOKE_DURATION_S",
    "S1CampaignExactAdvanceAdapter",
    "S1CampaignRuntimeBundle",
    "S1CampaignRuntimeError",
    "S1CoupledStateCodec",
    "S1Stage1ObservationBridge",
    "build_current_s1_campaign_runtime",
]
