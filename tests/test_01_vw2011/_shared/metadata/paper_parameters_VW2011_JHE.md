# Vasconcelos & Wright (2011), JHE 137(5):543–555 — apparatus & experimental parameters
Source: papers/vasconcelos2011.pdf (parsed). This is the ACTUAL paper being reproduced.

## Apparatus (Fig. 2) — horizontal pipe D = 0.094 m throughout
| component | length | note |
|---|---|---|
| upstream pipe | 0.546 m | holds the pressurized AIR (air pump + differential manometer); this volume = the air pocket |
| butterfly valve | D=0.102 m | between upstream and middle pipe; opened (<1 s) to start the run |
| middle pipe | 2.970 m | valve -> PVC coupling (where the ventilation tower sits) |
| ventilation tower | L = 0.610 m | variable diameter; open top; partially water-filled (initial level set) |
| downstream pipe | 0.490 m | connected to the coupling; **closed** at the downstream end |
| pressure transducer | — | 1.07 m downstream of the butterfly valve |

Tower junction location from the upstream (closed) end = 0.546 + 2.970 = **3.516 m**.
Total horizontal length = 0.546 + 2.970 + 0.490 = **4.006 m** (both far ends closed).

## Experimental variables (Table 1) — 36 configurations (4 x 3 x 3)
| variable | values |
|---|---|
| ventilation tower diameter D_t | 57.1, 44.4, 25.4, 12.7 mm  (D_t/D = 0.607, 0.472, 0.270, 0.135) |
| air-phase initial pressure head H_a0 | 0.305, 0.610, 0.915 m |
| initial water level in tower Y_fs0 | 0.254, 0.356, 0.457 m |

## Other fixed conditions
- Initial air volume = volume of the upstream pipe behind the valve = 0.546 m x (pi/4 x 0.094^2) = 3.79e-3 m^3 (3.79 L). (Reducing it to 25% was tried then dropped as a variable.)
- Initial air absolute pressure Pa0 = P_atm + rho_w g H_a0.
- Downstream pipe + middle pipe initially water-filled; tower water-filled to Y_fs0; tower top open.
- Polytropic/isentropic coefficient gamma = 1.4; material roughness eps/D_t = 0.0015.
- Initial air-water interface velocity in tower = Taylor bubble U_inf = 0.345 sqrt(g D_t).
- "Geysering" = the tower free surface Y_fs reaches the tower top L (water spills) before the
  rising air-pocket nose Y_int catches the free surface.

## Key experimental outcomes (to compare against)
- Tower diameter is the controlling variable.
  - D_t/D = 0.607: free-surface rise < 10% of L (no geyser).
  - D_t/D = 0.472: rise ~ 20% of L.
  - D_t/D = 0.270: rise up to ~40% of L (geyser in some runs).
  - D_t/D = 0.135: free surface always reaches the top (geyser every run), monotonic accelerating rise.
- Averaged non-dimensional upward velocities (Table 2), normalized by sqrt(g D_t):
  | D_t/D | mean V_fs | mean V_int |
  |---|---|---|
  | 0.607 | 0.048 | 0.39 |
  | 0.472 | 0.088 | 0.45 |
  | 0.270 | 0.30  | 0.79 |
  | 0.135 | 0.44  | 1.43 |
- Initial air pressure head and initial tower water level had only secondary effect (they set the
  starting free-surface level); higher initial water level -> more likely to geyser.
