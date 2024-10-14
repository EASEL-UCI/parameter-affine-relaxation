import numpy as np
import casadi as cs

from par.koopman import attitude_observables, gravity_observables
from par.constants import GRAVITY


v0 = cs.SX.sym("v0", 3)
J = 2.0 * np.eye(3)
Nv = 5

vs = attitude_observables.get_k_angular_velocities(v0, J, Nv)
Hs = attitude_observables.get_k_Hs(vs, J)
B = attitude_observables.get_input_matrix(Hs, J)

Ng = 3
vs_Ng = attitude_observables.get_k_angular_velocities(v0, J, Ng)
Hs_Ng = attitude_observables.get_k_Hs(vs_Ng, J)

g0 = cs.SX(cs.vertcat(0, 0, -GRAVITY))
gs = gravity_observables.get_k_gs(g0, vs_Ng)
Gs = gravity_observables.get_k_Gs(gs, vs_Ng, Hs_Ng, J)

print(Hs, "\n")
print(Gs)
