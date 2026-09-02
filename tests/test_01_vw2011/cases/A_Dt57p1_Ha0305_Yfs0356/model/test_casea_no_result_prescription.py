"""Guard Case A against result-prescribing closures."""

from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
PRODUCTION_SOURCES = (
    MODEL_DIR / "vw2011_network_twofluid.py",
    MODEL_DIR / "casea_shockfit_network.py",
    MODEL_DIR / "casea_postarrival_twofluid.py",
    MODEL_DIR / "casea_bidirectional_tnode_inertance.py",
    MODEL_DIR / "casea_tnode_mouth_phase_area.py",
    MODEL_DIR / "casea_distributed_tnode_inertance.py",
    MODEL_DIR / "casea_vertical_mouth_twochannel.py",
    MODEL_DIR / "casea_vertical_mouth_twochannel_integration.py",
    MODEL_DIR / "casea_vertical_twostream_closures.py",
    MODEL_DIR / "casea_vertical_twostream_fv.py",
    MODEL_DIR / "casea_vertical_bottom_riemann.py",
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
    # After the fitted front reaches the T, the distributed horizontal gas
    # field is the sole owner.  Reintroducing either token would restore the
    # rejected parallel lumped-reservoir path and suppress resolved waves.
    "advance_lumped_pocket_vertical_network",
    "external_open_gas_inventory",
    # The rejected viewer used a fitted Taylor-front sweep and a symmetric
    # axial impulse to manufacture horizontal waves at the side tee.  The
    # production graph must obtain those waves from its conservative fluxes.
    "_sweep_vertical_material_slice_to_junction",
    "incoming_normal_velocity",
)


def test_production_sources_contain_no_result_prescription() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in PRODUCTION_SOURCES
    )
    present = [token for token in BANNED_TOKENS if token in combined]
    assert not present, f"result-prescribing closures returned: {present}"
