#!/bin/bash
source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
cd /mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d
foamToVTK -fields '(alpha.water)' -time '0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.4,2.55,2.7,2.85,3,3.15,3.3,3.45,3.6,3.75,3.9,4.05,4.2,4.35,4.55,4.7,4.85,5,5.15,5.3,5.45,5.6,5.75,5.9,6.05,6.2,6.35,6.5,6.65,6.8,6.95,7.1,7.25,7.4,7.55,7.7,7.85,8,8.15,8.3,8.45,8.6,8.75,8.95' -ascii > log.foamToVTK_compare 2>&1
