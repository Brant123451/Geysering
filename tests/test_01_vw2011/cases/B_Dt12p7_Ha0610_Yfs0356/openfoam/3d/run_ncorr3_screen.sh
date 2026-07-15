#!/bin/bash
# Launch wrapper for the nCorrectors=3 rim-onset screen.
# Allrun sources OpenFOAM itself; do not source bashrc here.
set -euo pipefail
cd "$(dirname "$0")"

export CASEB_STAGE=hold
export CASEB_MESH=base
export CASEB_VALVE_MODE=closed
export CASEB_END_TIME=0.12
export CASEB_WRITE_INTERVAL=0.01
export CASEB_MAX_CO=0.15
export CASEB_MAX_ALPHA_CO=0.2
export CASEB_MAX_CAPILLARY_NUM=1.0
export CASEB_MAX_DELTA_T=0.00025
export CASEB_N_CORRECTORS=3
export CASEB_N_OUTER_CORRECTORS=1
export CASEB_N_ALPHA_CORR=1
export CASEB_N_ALPHA_SUBCYCLES=2
export CASEB_N_NON_ORTHOGONAL_CORRECTORS=0
export CASEB_PRESSURE_FINAL_TOLERANCE=1e-10
export CASEB_HYDROSTATIC_INITIALIZATION=discrete
export CASEB_HYDROSTATIC_CORRECTORS=10
export CASEB_INTERPOLATE_NORMAL=false
export CASEB_CURVATURE_MODEL=RDF
export CASEB_CURV_FROM_TR=true
export CASEB_RECONSTRUCTION_SCHEME=plicRDF
export CASEB_RECONSTRUCTION_ITERATIONS=5
export CASEB_RECONSTRUCTION_TOL=1e-6
export CASEB_ADVECTION_SCHEME=isoAdvection
export OPENFOAM_NP=4

echo "START_NCORR3 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
./Allclean
./Allrun
echo "END_NCORR3 exit:$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
