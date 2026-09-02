#!/bin/bash
cd "$(dirname "$0")"
echo "=== latest time/courant ==="
grep -E '^(Time =|Courant Number|deltaT =)' log.compressibleInterFlow | tail -30
echo "=== hotspot / bounds ==="
grep -E 'CASEB_|y=0\.65|Umax' log.compressibleInterFlow | tail -30
