import pytest

from model import (
    AtomicCommitter,
    CoupledStepper,
    GasSourceContext,
    MissingPhysicalClosure,
    UnresolvedPressureGasSource,
)


def test_published_pressure_is_not_silently_converted_to_mass_flow() -> None:
    source = UnresolvedPressureGasSource(source_gauge_pressure_Pa=5700.0)
    context = GasSourceContext(
        time_s=0.0,
        dt_s=0.01,
        local_absolute_pressure_Pa=101325.0,
        local_gas_density_kg_m3=1.2,
    )
    with pytest.raises(MissingPhysicalClosure, match="not a gas mass-flow prescription"):
        source.evaluate(context)


def test_coupled_stepper_refuses_to_make_a_trajectory_without_closures(
    coupled_state, geometry
) -> None:
    stepper = CoupledStepper(AtomicCommitter(geometry))
    with pytest.raises(MissingPhysicalClosure, match="refuses to generate a trajectory"):
        stepper.advance(coupled_state, dt_s=0.01)
