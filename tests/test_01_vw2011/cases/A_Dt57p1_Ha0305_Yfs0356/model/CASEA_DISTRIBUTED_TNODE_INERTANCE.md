# Case-A distributed finite-width T-node inertance

`casea_distributed_tnode_inertance.py` is an isolated, target-free owner for
the signed liquid flux through the horizontal-pipe/riser mouth. It does not
modify `vw2011_network_twofluid.py` and is not active in production until the
main loop explicitly persists its momentum state and disables every legacy
mouth-flux owner.

## State and geometry

The persistent state is the moving liquid momentum

\[
P_J=\rho_l L_{\mathrm{eff}}Q_J,
\qquad
L_{\mathrm{eff}}=V_{\mathrm{footprint}}/A_r .
\]

For Case A, the geometry constructor uses the measured horizontal diameter
`D = 0.094 m`, riser diameter `Dr = 0.0571 m`, physical opening length
`L_open = Dr`, and footprint volume `V_footprint = A_pipe L_open`. A grid
adapter may supply an exactly integrated physical-overlap volume, but must not
replace the measured opening length with a smoothing or display width.

The horizontal driving pressure is the current gas/liquid contact-area
average in the footprint. The opposing pressure is the current resolved
vertical mouth pressure. No elapsed time, 2-D field, target hold-up, target
flux, or prescribed entry pulse is accepted by the API.

## Pressure/flux solve

Each step solves the monotone implicit relation

\[
I_p\frac{Q_J^{n+1}-Q_J^n}{\Delta t}
+\frac{\rho_l K_{\pm}}{2A_r^2}Q_J^{n+1}|Q_J^{n+1}|
=\Delta p+\lambda_- -\lambda_+,
\]

where `I_p = rho_l L_eff / A_r`; `K+` and `K-` are the directional local
turn losses. The total downward Nusselt/Wallis capacity is a lower flux bound,
not a post-solve limiter. If it is active, the returned ledger contains its
non-negative reaction pressure, pressure impulse, feasibility gap,
complementarity product, pressure-balance residual, and momentum-balance
residual. Horizontal/riser donor positivity uses the same explicit
box-complementarity machinery and has separate reaction fields.

## Gross exchange and conservation

Only after the net-flux solve passes does the component call
`stage_twochannel_mouth_coupling`. That existing closure returns simultaneous
gross `Q_up` and `Q_down` while preserving

\[
Q_{up}-Q_{down}=Q_J.
\]

The component then applies `Q_J` exactly once with opposite signs to the
finite horizontal node and riser lumped inventories. The returned combined
volume residual must be roundoff zero. Gross circulation is not inserted into
a single-momentum riser; production integration requires the persistent
two-stream riser state.

## Main-loop integration contract

At the first physically connected mouth-topology event (the horizontal
junction contains gas while the riser mouth is physically able to receive
that gas), the main solver must act. This event precedes and must not be
confused with gas reaching the riser free surface. The main solver must:

1. initialise and then persist `DistributedTNodeMomentumState`;
2. supply current-stage horizontal gas/liquid pressures, vertical mouth
   pressure, phase areas/velocities, and donor inventories;
3. use this result as the sole signed mouth-flux owner;
4. disable the applied characteristic bottom flux, Taylor return mass flux,
   post-breakthrough CCFL clipping of signed flow, and old net-only side
   source;
5. pass the returned gross boundary streams to the persistent two-stream
   riser and apply the horizontal footprint update once; and
6. reject the step if any pressure, momentum, complementarity, phase, or
   combined-volume ledger fails.

The current opening weights are normalized physical overlaps,
`weight_i = overlap_i / L_open`. Therefore the horizontal donor inventory is
`sum(weight_i * A_i) * L_open`, not `sum(weight_i * A_i) * dx`. The helper
`measured_footprint_liquid_inventory` implements the correct integral. The
gross footprint application must use the same physical overlap scale; using
`dx` is grid-dependent unless `dx == L_open`.

Run the isolated checks with:

```powershell
python -m pytest model/test_casea_distributed_tnode_inertance.py -q
```
