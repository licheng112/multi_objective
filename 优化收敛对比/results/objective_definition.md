# Objective Definition

F1(x) = |Y_IV - 50|
F2(x) = |Y_DEG - 1.37|
F3(x) = |Y_CTA - 51|
Raw average objective value = (F1 + F2 + F3) / 3
Objective values are robust to process disturbances: mean absolute deviation over disturbance scenarios plus a weighted deviation term.
Candidates outside prior process ranges receive larger disturbance amplitudes to represent lower process stability in abnormal operating regions.
A parameter-adjustment stability term is included so that large departures from the historical stable operating center are penalized.
Average objective value in convergence curves is constraint-aware: robust target deviations plus prior-constraint violation penalty.

HV and IGD are calculated in normalized constraint-aware objective space using the historical 95%-5% ranges of IV, DEG, and CTA.
