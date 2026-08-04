# Selected V&W(2011) Test Reproduction

This folder is now scoped to one representative test from Vasconcelos & Wright
(2011), JHE 137(5):543-555.

## Selected test

| item | value |
|---|---:|
| Horizontal pipe diameter, D | 0.094 m |
| Upstream air chamber length | 0.546 m |
| Middle water-filled pipe length | 2.970 m |
| Ventilation tower location from upstream end | 3.516 m |
| Ventilation tower length, L | 0.610 m |
| Downstream closed pipe length | 0.490 m |
| Ventilation tower diameter, Dt | 0.0127 m |
| Initial air pressure head, Ha0 | 0.610 m |
| Initial tower water level, Yfs0 | 0.356 m |
| Gas exponent, gamma | 1.4 |

## Reproduction rule

The simulation uses the selected paper geometry and initial condition, then lets
the decoupled two-fluid model from `D:\tests\Research\论文\mixed_flow_submission_en_clean_20260526_1930`
evolve the flow. The geyser motion is not prescribed by a release-rate script,
overflow accumulator, or fitted target height.

## Output

Run:

```powershell
python .\vw2011_network_twofluid.py --mode report --regen
```

The manual frame-viewer HTML is written to:

```text
outputs\vw2011_network\report.html
```

It contains only the selected simulation frame sequence with a slider, previous/next
buttons, and play/pause controls.
