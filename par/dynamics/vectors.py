from typing import Union, List

import numpy as np
import casadi as cs

from par.utils import quat, math
from par.utils.misc import is_none, convert_casadi_to_numpy_vector
from par.utils.config import get_dimensions, get_config_values
from par.koopman.observables import attitude, gravity, velocity, position
from par.constants import GRAVITY
from par.config import PARAMETER_CONFIG, RELAXED_PARAMETER_CONFIG, \
                        STATE_CONFIG, KOOPMAN_CONFIG, INPUT_CONFIG


class DynamicsVector():
    def __init__(
        self,
        vector: np.ndarray,
        config: dict,
    ) -> None:
        self._dims = get_dimensions(config)
        self._config = config
        self._members = {}
        self.set_vector(vector)

    def as_array(self) -> np.ndarray:
        vector = [member for member in self._members.values()]
        return np.hstack(vector)

    def as_list(self) -> np.ndarray:
        return list(self.as_array())

    def set_vector(self, vector: np.ndarray) -> None:
        assert len(vector) == self._dims
        i = 0
        for id, subconfig in self._config.items():
            dims = subconfig["dimensions"]
            self._members[id] = vector[i : i + dims]
            i += dims

    def get_member(self, id: str) -> Union[float, np.ndarray]:
        return self._members[id]

    def set_member(
        self,
        id: str,
        member: Union[float, np.ndarray],
    ) -> None:
        if type(member) == float or type(member) == np.float64:
            assert self._config[id]["dimensions"] == 1
        else:
            assert len(member) == self._config[id]["dimensions"]
        self._members[id] = member


class DynamicsVectorList(list):
    def __init__(
        self,
        vector_list=[]
    ) -> None:
        self._list = []
        self.append(vector_list)

    def as_array(self) -> np.ndarray:
        return np.array( [vec.as_array() for vec in self._list] )

    def get(
        self,
        index=None
    ) -> DynamicsVector:
        if is_none(index):
            return self._list
        else:
            return self._list[index]

    def append(
        self,
        vectors: Union[DynamicsVector, List[DynamicsVector]]
    ) -> None:
        if type(vectors) == DynamicsVector or type(vectors) == ModelParameters \
        or type(vectors) == State or type(vectors) == Input:
            self._assert_type(vectors)
            self._list += [vectors]
        else:
            map(self._assert_type, vectors)
            self._list += vectors

    def _assert_type(
        self,
        entry: DynamicsVector
    ) -> None:
        assert type(entry) == DynamicsVector or type(entry) == ModelParameters \
            or type(entry) == State or type(entry) == Input


class Input(DynamicsVector):
    def __init__(
        self,
        u=None
    ) -> None:
        if is_none(u):
            u = get_config_values("default_value", INPUT_CONFIG)
        super().__init__(u, INPUT_CONFIG)


class State(DynamicsVector):
    def __init__(
        self,
        x=None,
    ) -> None:
        if is_none(x):
            x = get_config_values("default_value", STATE_CONFIG)
        super().__init__(x, STATE_CONFIG)

    def get_zero_order_koopman_vector(self) -> np.ndarray:
        z0_members = self.get_zero_order_koopman_members()
        z0 = [list(z0_members[id]) for id in KOOPMAN_CONFIG.keys()]
        return np.hstack(z0).flatten()

    def get_lifted_koopman_vector(self, J: np.ndarray, order: int) -> np.ndarray:
        z_members = self.get_lifted_koopman_members(J, order)
        z = [list(z_members[id]) for id in KOOPMAN_CONFIG.keys()]
        return np.hstack(z).flatten()

    def get_zero_order_koopman_members(self) -> dict:
        rot = quat.Q(self._members["ATTITUDE"])
        z0_members = {}
        z0_members["BODY_FRAME_POSITION"] = rot.T @ self._members["POSITION"]
        z0_members["BODY_FRAME_LINEAR_VELOCITY"] = \
                                        self._members["BODY_LINEAR_VELOCITY"]
        z0_members["BODY_FRAME_GRAVITY"] = rot.T @ (-GRAVITY * math.e3())
        z0_members["BODY_FRAME_ANGULAR_VELOCITY"] = \
                                        self._members["BODY_ANGULAR_VELOCITY"]
        return z0_members

    def get_lifted_koopman_members(self, J: np.ndarray, order: int) -> dict:
        z0_members = self.get_zero_order_koopman_members()
        ws = attitude.get_ws(
            z0_members["BODY_FRAME_ANGULAR_VELOCITY"], J, order)
        ps = position.get_ps(z0_members["BODY_FRAME_POSITION"], ws)
        vs = velocity.get_vs(z0_members["BODY_FRAME_LINEAR_VELOCITY"], ws)
        gs = gravity.get_gs(z0_members["BODY_FRAME_GRAVITY"], ws)

        ps_vec = convert_casadi_to_numpy_vector(cs.vertcat(*ps))
        vs_vec = convert_casadi_to_numpy_vector(cs.vertcat(*vs))
        gs_vec = convert_casadi_to_numpy_vector(cs.vertcat(*gs))
        ws_vec = convert_casadi_to_numpy_vector(cs.vertcat(*ws))
        z_members = {}
        z_members["BODY_FRAME_POSITION"] = ps_vec
        z_members["BODY_FRAME_LINEAR_VELOCITY"] = vs_vec
        z_members["BODY_FRAME_GRAVITY"] = gs_vec
        z_members["BODY_FRAME_ANGULAR_VELOCITY"] = ws_vec
        return z_members


class ModelParameters(DynamicsVector):
    def __init__(
        self,
        theta=None,
    ) -> None:
        if is_none(theta):
            theta = get_config_values("default_value", PARAMETER_CONFIG)
        super().__init__(theta, PARAMETER_CONFIG)

    def get_affine_vector(self) -> np.ndarray:
        aff_members = self.get_affine_members()
        theta_aff = \
            [list(aff_members[id]) for id in RELAXED_PARAMETER_CONFIG.keys()]
        return np.hstack(theta_aff).flatten()

    def get_affine_members(self) -> dict:
        m = self._members["m"]
        k = self._members["k"]
        Ixx = self._members["Ixx"]
        Iyy = self._members["Iyy"]
        Izz = self._members["Izz"]
        aff_members = {}
        aff_members["A"] = self._members["a"] / m
        aff_members["K"] = k / m
        aff_members["S"] = k * self._members["s"] / Ixx
        aff_members["R"] = k * self._members["r"] / Iyy
        aff_members["C"] = self._members["c"] / Izz
        aff_members["IXX"] = (Izz - Iyy) / Ixx
        aff_members["IYY"] = (Ixx - Izz) / Iyy
        aff_members["IZZ"] = (Iyy - Ixx) / Izz
        return aff_members
