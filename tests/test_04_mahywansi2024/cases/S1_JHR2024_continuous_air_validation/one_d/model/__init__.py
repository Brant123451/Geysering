"""Contract-first 1-D coupling skeleton for the S1 validation case.

This package contains source-aligned, fail-closed component closures but no
accepted production trajectory yet.  It locks down state ownership, atomic
two-T-node exchange, conservation accounting and the riser-mouth event
definition so that Case-1 operator reuse cannot silently change those contracts.
"""

from .boundaries import (
    ContinuousGasSourceBoundary,
    GasSourceContext,
    GasSourceFlux,
    UnresolvedPressureGasSource,
)
from .conservation import ConservationLedger, ConservationSnapshot, LedgerEntry
from .coupled import AtomicCommitter, AtomicFluxClosure, CoupledStepper
from .errors import (
    AtomicCommitError,
    ConservationError,
    ContractViolation,
    MissingPhysicalClosure,
)
from .events import TopOutflowEventIntegrator, TopOutflowEventSnapshot
from .atmospheric_exterior_plume import (
    ExteriorPlumeStageDiagnostics,
    ExteriorPlumeStageEvaluation,
    F0AtmosphericExteriorPlumeOwner,
)
from .flux import (
    AtomicFluxPacket,
    BoundaryExchange,
    ExteriorPlumeDelta,
    HorizontalDelta,
    SupplyBranchDelta,
    TNodeDelta,
    TNodePortResidual,
    VerticalDelta,
)
from .horizontal_case1_adapter import (
    FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S,
    build_s1_2d_eos_aligned_horizontal_adapter,
)
from .horizontal_two_tee_component import (
    F0HorizontalTwoTeeStageComponent,
    HorizontalF0Readiness,
)
from .port_contracts import (
    CapacityReject,
    CapillaryGeometryMode,
    CapillaryInterfaceOwnership,
    ComponentStageProposal,
    GrossNodePortFlux,
    PortKey,
    PortTraceState,
    TNodeTrial,
)
from .physical_joint_owner import (
    F0PhysicalTwoTNodeStageOwner,
    PhysicalJointStageInputs,
)
from .state import (
    CoupledGeometry,
    CoupledState,
    ExteriorPlumeState,
    HorizontalState,
    SupplyBranchState,
    TNodeState,
    VerticalState,
)
from .vertical_pressure_void_component import (
    AtmosphericLiquidFlux,
    AtmosphericLiquidFallback,
    AtmosphericTopState,
    BottomGasPistonRemap,
    F0VerticalCapillaryOwner,
    F0VerticalPressureVoidStageComponent,
    conservative_void_remap,
)

__all__ = [
    "AtomicCommitError",
    "AtomicCommitter",
    "AtomicFluxClosure",
    "AtomicFluxPacket",
    "BoundaryExchange",
    "ExteriorPlumeDelta",
    "ExteriorPlumeStageDiagnostics",
    "ExteriorPlumeStageEvaluation",
    "ExteriorPlumeState",
    "F0AtmosphericExteriorPlumeOwner",
    "CapacityReject",
    "CapillaryGeometryMode",
    "CapillaryInterfaceOwnership",
    "ComponentStageProposal",
    "ConservationError",
    "ConservationLedger",
    "ConservationSnapshot",
    "ContractViolation",
    "ContinuousGasSourceBoundary",
    "CoupledGeometry",
    "CoupledState",
    "CoupledStepper",
    "GasSourceContext",
    "GasSourceFlux",
    "F0HorizontalTwoTeeStageComponent",
    "F0PhysicalTwoTNodeStageOwner",
    "F0VerticalCapillaryOwner",
    "F0VerticalPressureVoidStageComponent",
    "FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S",
    "GrossNodePortFlux",
    "HorizontalDelta",
    "HorizontalF0Readiness",
    "HorizontalState",
    "LedgerEntry",
    "MissingPhysicalClosure",
    "PortKey",
    "PortTraceState",
    "PhysicalJointStageInputs",
    "SupplyBranchDelta",
    "SupplyBranchState",
    "TNodeDelta",
    "TNodePortResidual",
    "TNodeState",
    "TNodeTrial",
    "TopOutflowEventIntegrator",
    "TopOutflowEventSnapshot",
    "UnresolvedPressureGasSource",
    "VerticalDelta",
    "VerticalState",
    "AtmosphericLiquidFlux",
    "AtmosphericLiquidFallback",
    "AtmosphericTopState",
    "BottomGasPistonRemap",
    "build_s1_2d_eos_aligned_horizontal_adapter",
    "conservative_void_remap",
]
