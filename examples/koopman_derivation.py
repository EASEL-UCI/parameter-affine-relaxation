import numpy as np
import casadi as cs

from par.koopman.observables import attitude, gravity, velocity, position
from par.constants import GRAVITY


J = 2.0 * np.eye(3)


NV = 3
angular_velocity_0 = cs.SX.sym("angular_velocity_0", 3)
angular_velocities = attitude.get_angular_velocities(
    angular_velocity_0, J, NV)
Hs = attitude.get_Hs(angular_velocities, J)
B = attitude.get_input_matrix(Hs, J)
print(f"{len(Hs)} x {Hs[-1].shape}  \n")


Ng = 4
angular_velocities_Ng = attitude.get_angular_velocities(
    angular_velocity_0, J, Ng)
Hs_Ng = attitude.get_Hs(angular_velocities_Ng, J)

g0 = cs.SX(cs.vertcat(0, 0, -GRAVITY))
gs = gravity.get_gs(g0, angular_velocities_Ng)
Gs = gravity.get_Gs(gs, angular_velocities_Ng, Hs_Ng, J)
print(f"{len(Gs)} x {Gs[-1].shape}  \n")


Nv = 5
angular_velocities_Nz = attitude.get_angular_velocities(
    angular_velocity_0, J, Nv)
Hs_Nz = attitude.get_Hs(angular_velocities_Nz, J)

velocity_0 = cs.SX.sym("velocity_0", 3)
velocities = velocity.get_velocities(
    velocity_0, angular_velocities_Nz)
omegas = velocity.get_omegas(velocities)
Vs = velocity.get_Vs(velocities, angular_velocities_Nz, Hs_Nz, J)
print(f"{len(Vs)} x {Vs[-1].shape}  \n")


Np = 6
angular_velocities_Np = attitude.get_angular_velocities(
    angular_velocity_0, J, Np)
Hs_Np = attitude.get_Hs(angular_velocities_Np, J)

position_0 = cs.SX.sym("position_0", 3)
positions = position.get_positions(
    position_0, angular_velocities_Np)
Ps = position.get_Ps(positions, angular_velocities_Np, Hs_Np, J)
print(f"{len(Ps)} x {Ps[-1].shape}  \n")
