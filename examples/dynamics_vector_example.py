import numpy as np

from par.dynamics.vectors import State, ModelParameters


x = State()
x.set_vector(np.zeros(13))
x.set_member("ATTITUDE", np.hstack((1.0, np.zeros(3))))
print(x.get_vector())
print(x.get_zero_order_koopman_vector())
print(x.get_lifted_koopman_vector(2*np.eye(3), order=3))
print()


theta = ModelParameters()
theta.set_vector(np.ones(23))
theta.set_member("Ixx", 100.0)
print(theta.get_vector())
print(theta.get_affine_vector())
