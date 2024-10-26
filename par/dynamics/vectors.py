from typing import Union, List

import numpy as np
import casadi as cs

from par.utils import quat, math
from par.utils.misc import is_none, convert_casadi_to_numpy_array
from par.utils.config import get_dimensions, get_config_values
from par.koopman.observables import attitude, gravity, velocity, position
from par.constants import GRAVITY
from par.config import PARAMETER_CONFIG, RELAXED_PARAMETER_CONFIG, PROCESS_NOISE_CONFIG,\
                        STATE_CONFIG, KOOPMAN_STATE_CONFIG, INPUT_CONFIG


class DynamicsVector():
    def __init__(
        self,
        config: dict,
        array: np.ndarray = None,
        copies: int = 1,
    ) -> None:
        self._n = copies
        self._dims = get_dimensions(config)
        self._config = config
        self._members = {}
        if is_none(array):
            array = get_config_values('default_value', config, copies=copies)
        self.set(array)

    @property
    def config(self) -> dict:
        return self._config

    def as_array(self) -> np.ndarray:
        nonflat_list = [member for member in self._members.values()]
        return np.hstack(nonflat_list)

    def as_list(self) -> List:
        return list(self.as_array())

    def set(self, vector: np.ndarray) -> None:
        assert len(vector) == self._n * self._dims
        i = 0
        for id, subconfig in self._config.items():
            dims = self._n * subconfig['dimensions']
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
            assert self._config[id]['dimensions'] == self._n
        else:
            assert len(member) == self._n * self._config[id]['dimensions']
        self._members[id] = member


class VectorList():
    def __init__(
        self,
        vector_list: List = [],
    ) -> None:
        self._list = []
        self.append(vector_list)

    def as_array(self) -> np.ndarray:
        return np.array( [vec.as_array() for vec in self._list] )

    def get(
        self,
        index: int = None,
    ) -> DynamicsVector:
        if is_none(index):
            return self._list
        else:
            return self._list[index]

    def set(
        self,
        index: int,
        vectors: Union[DynamicsVector, List[DynamicsVector]],
    ) -> None:
        self._assert_type(vectors)
        self._list[index] = vectors

    def pop(
        self,
        index: int,
    ) -> DynamicsVector:
        return self._list.pop(index)

    def append(
        self,
        vectors: Union[DynamicsVector, List[DynamicsVector]],
    ) -> None:
        if type(vectors) == list:
            map(self._assert_type, vectors)
            self._list += vectors
        elif self._check_type(vectors):
            self._assert_type(vectors)
            self._list += [vectors]
        elif type(vectors) == list:
            map(self._assert_type, vectors)
            self._list += vectors
        else:
            raise TypeError('Attempted to append invalid type!')

    def _assert_type(
        self,
        entry: DynamicsVector,
    ) -> None:
        assert self._check_type(entry)

    def _check_type(
        self,
        entry: DynamicsVector,
    ) -> bool:
        valid_types = {
            DynamicsVector: DynamicsVector,
            ModelParameters: ModelParameters,
            State: State,
            Input: Input,
            KoopmanLiftedState: KoopmanLiftedState,
            AffineModelParameters: AffineModelParameters,
        }
        try:
            valid_types[type(entry)]
        except KeyError:
            return False
        except TypeError:
            return False
        finally:
            return True


class Input(DynamicsVector):
    def __init__(
        self,
        u: np.ndarray = None,
    ) -> None:
        super().__init__(INPUT_CONFIG, u)


class KoopmanLiftedState(DynamicsVector):
    def __init__(
        self,
        z: np.ndarray = None,
        order: int = 1,
    ) -> None:
        super().__init__(KOOPMAN_STATE_CONFIG, z, order)

    def get_zero_order_array(self) -> np.ndarray:
        vector = [member[: self._config[id]['dimensions']] \
                    for id, member in self._members.items()]
        return np.hstack(vector)


class State(DynamicsVector):
    def __init__(
        self,
        x: np.ndarray = None,
    ) -> None:
        super().__init__(STATE_CONFIG, x)

    def as_zero_order_koopman(self) -> KoopmanLiftedState:
        z0_members = self.get_zero_order_koopman_members()
        z0 = [z0_members[id] for id in KOOPMAN_STATE_CONFIG.keys()]
        return KoopmanLiftedState(np.hstack(z0).flatten(), 1)

    def as_lifted_koopman(self, J: np.ndarray, order: int) -> KoopmanLiftedState:
        z_members = self.get_lifted_koopman_members(J, order)
        z = [z_members[id] for id in KOOPMAN_STATE_CONFIG.keys()]
        return KoopmanLiftedState(np.hstack(z).flatten(), order)

    def get_zero_order_koopman_members(self) -> dict:
        rot = quat.Q(self._members['attitude'])
        z0_members = {}
        z0_members['position_bf'] = rot.T @ self._members['position_wf']
        z0_members['linear_velocity_bf'] = \
            self._members['linear_velocity_bf']
        z0_members['gravity_bf'] = rot.T @ (-GRAVITY * math.e3())
        z0_members['angular_velocity_bf'] = \
            self._members['angular_velocity_bf']
        return z0_members

    def get_lifted_koopman_members(self, J: np.ndarray, order: int) -> dict:
        z0_members = self.get_zero_order_koopman_members()
        ws = attitude.get_ws(
            z0_members['angular_velocity_bf'], J, order)
        ps = position.get_ps(z0_members['position_bf'], ws)
        vs = velocity.get_vs(z0_members['linear_velocity_bf'], ws)
        gs = gravity.get_gs(z0_members['gravity_bf'], ws)

        ps_vec = convert_casadi_to_numpy_array(cs.vertcat(*ps))
        vs_vec = convert_casadi_to_numpy_array(cs.vertcat(*vs))
        gs_vec = convert_casadi_to_numpy_array(cs.vertcat(*gs))
        ws_vec = convert_casadi_to_numpy_array(cs.vertcat(*ws))
        z_members = {}
        z_members['position_bf'] = ps_vec
        z_members['linear_velocity_bf'] = vs_vec
        z_members['gravity_bf'] = gs_vec
        z_members['angular_velocity_bf'] = ws_vec
        return z_members


class AffineModelParameters(DynamicsVector):
    def __init__(
        self,
        theta_aff: np.ndarray = None,
    ) -> None:
        super().__init__(RELAXED_PARAMETER_CONFIG, theta_aff)


class ModelParameters(DynamicsVector):
    def __init__(
        self,
        theta: np.ndarray = None,
    ) -> None:
        super().__init__(PARAMETER_CONFIG, theta)

    def as_affine(self) -> AffineModelParameters:
        aff_members = self.get_affine_members()
        theta_aff = \
            [aff_members[id] for id in RELAXED_PARAMETER_CONFIG.keys()]
        return AffineModelParameters(np.hstack(theta_aff).flatten())

    def get_affine_members(self) -> dict:
        m = self._members['m']
        Ixx = self._members['Ixx']
        Iyy = self._members['Iyy']
        Izz = self._members['Izz']

        aff_members = {}
        aff_members['M'] = 1 / m
        aff_members['A'] = self._members['a'] / m
        aff_members['IXX'] = 1 / Ixx
        aff_members['IYY'] = 1 / Iyy
        aff_members['IZZ'] = self._members['b'] / Izz
        aff_members['IXX_rb'] = (Izz - Iyy) / Ixx
        aff_members['IYY_rb'] = (Ixx - Izz) / Iyy
        aff_members['IZZ_rb'] = (Iyy - Ixx) / Izz
        return aff_members


class ProcessNoise(DynamicsVector):
    def __init__(
        self,
        w: np.ndarray = None,
    ) -> None:
        super().__init__(PROCESS_NOISE_CONFIG, w)
