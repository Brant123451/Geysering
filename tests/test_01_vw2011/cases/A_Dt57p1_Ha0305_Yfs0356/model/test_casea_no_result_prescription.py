"""Guard Case A against result-prescribing closures."""

from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
PRODUCTION_SOURCES = (
    MODEL_DIR / "vw2011_network_twofluid.py",
    MODEL_DIR / "casea_shockfit_network.py",
    MODEL_DIR / "casea_postarrival_twofluid.py",
)

BANNED_TOKENS = (
    "_hold_at_side_t",
    "side_t_projection_volume",
    "advance_downstream_dead_leg",
    "tower_entry_delay_s",
    "tower_entry_elapsed",
    "vent_latched",
    "pocket_bleed",
    "film_span_overlay",
    "_project_vented_connected_horizontal_pocket",
    "_rescale_connected_void_profile",
    "target_jet_height",
    "pressure_launch_velocity",
    # The Nusselt/Davies--Taylor value is a diagnostic/wall-shear scale, not a
    # second liquid boundary condition at the shared T face.
    "taylor_film_return",
    "film_momentum_flux = film_flow",
)


def test_production_sources_contain_no_result_prescription() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in PRODUCTION_SOURCES
    )
    present = [token for token in BANNED_TOKENS if token in combined]
    assert not present, f"result-prescribing closures returned: {present}"
