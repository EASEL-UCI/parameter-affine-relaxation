from typing import Union, List, Tuple

import numpy as np
import casadi as cs

from par.utils import quat, math
from par.utils.misc import is_none, convert_casadi_to_numpy_array
from par.utils.config import get_dimensions, get_config_values
from par.constants import GRAVITY
from par.config import *


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
            member = vector[i : i + dims]
            self.set_member(id, member)
            i += dims

    def get_member(self, id: str) -> Union[float, np.ndarray]:
        return self._members[id]

    def set_member(
        self,
        id: str,
        member: Union[float, np.ndarray],
    ) -> None:
        self._assert(id, member)
        member = np.clip(
            member,
            np.repeat(self._config[id]['lower_bound'], self._n),
            np.repeat(self._config[id]['upper_bound'], self._n)
        )
        self._members[id] = member

    def _assert(
        self,
        id: str,
        member: np.ndarray,
    ) -> None:
        try:
            if len(member) == 1:
                member = member[0]
        except TypeError:
            pass

        if type(member) == float or type(member) == np.float64:
            assert self._config[id]['dimensions'] == self._n
        else:
            assert len(member) == self._n * self._config[id]['dimensions']


class VectorList():
    def __init__(
        self,
        vector_list: List = [],
    ) -> None:
        self._list = []
        self.append(vector_list)

    def as_array(self) -> np.ndarray:
        return np.array( [vec.as_array() for vec in self._list] )

    def as_list(self) -> List:
        return list(self.as_array().flatten())

    def get(
        self,
        index: int = None,
    ) -> Union[DynamicsVector, List[DynamicsVector]]:
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


class State(DynamicsVector):
    def __init__(
        self,
        x: np.ndarray = None,
        order=1,
    ) -> None:
        super().__init__(STATE_CONFIG, x, order)


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
        theta_aff = [aff_members[id] for id in RELAXED_PARAMETER_CONFIG.keys()]
        return AffineModelParameters(np.hstack(theta_aff).flatten())

    def get_affine_members(self) -> dict:
        m = self._members['m']
        Ixx = self._members['Ixx']
        Iyy = self._members['Iyy']
        Izz = self._members['Izz']

        aff_members = {}
        aff_members['1/m'] = 1 / m
        aff_members['a/m'] = self._members['a'] / m
        aff_members['d/Ixx'] = self._members['d'] / Ixx
        aff_members['c/Iyy'] = self._members['c'] / Iyy
        aff_members['b/Izz'] = self._members['b'] / Izz
        aff_members['Ixx_cross'] = (Izz - Iyy) / Ixx
        aff_members['Iyy_cross'] = (Ixx - Izz) / Iyy
        aff_members['Izz_cross'] = (Iyy - Ixx) / Izz
        return aff_members


class ProcessNoise(DynamicsVector):
    def __init__(
        self,
        w: np.ndarray = None,
    ) -> None:
        super().__init__(PROCESS_NOISE_CONFIG, w)


def get_parameter_bounds(
    nominal_parameters: ModelParameters,
    lb_factor: float,
    ub_factor: float,
) -> Tuple[ModelParameters, ModelParameters]:
    lb_theta_arr = lb_factor * nominal_parameters.as_array()
    ub_theta_arr = ub_factor * nominal_parameters.as_array()
    return (
        ModelParameters(np.minimum(lb_theta_arr, ub_theta_arr)),
        ModelParameters(np.maximum(lb_theta_arr, ub_theta_arr))
    )


def get_affine_parameter_bounds(
    lb_theta: ModelParameters,
    ub_theta: ModelParameters,
) -> Tuple[AffineModelParameters, AffineModelParameters]:
    lb_theta = ModelParameters(np.minimum(lb_theta.as_array(), ub_theta.as_array()))
    ub_theta = ModelParameters(np.maximum(lb_theta.as_array(), ub_theta.as_array()))

    lb_theta_aff = AffineModelParameters()
    ub_theta_aff = AffineModelParameters()

    lb_theta_aff.set_member('1/m', 1 / ub_theta.get_member('m'))
    ub_theta_aff.set_member('1/m', 1 / lb_theta.get_member('m'))

    lb_theta_aff.set_member('a/m', lb_theta.get_member('a') / ub_theta.get_member('m'))
    ub_theta_aff.set_member('a/m', ub_theta.get_member('a') / lb_theta.get_member('m'))

    # s/Ixx bounds
    lb_s = lb_theta.get_member('d')
    ub_s = ub_theta.get_member('d')
    lb_s_aff = np.zeros(4)
    ub_s_aff = np.zeros(4)
    for i in range(len(lb_s)):
        if lb_s[i] >= 0:
            lb_s_aff[i] = lb_s[i] / ub_theta.get_member('Ixx')
        else:
            lb_s_aff[i] = lb_s[i] / lb_theta.get_member('Ixx')
        if ub_s[i] >= 0:
            ub_s_aff[i] = ub_s[i] / lb_theta.get_member('Ixx')
        else:
            ub_s_aff[i] = ub_s[i] / ub_theta.get_member('Ixx')
    lb_theta_aff.set_member('d/Ixx', lb_s_aff)
    ub_theta_aff.set_member('d/Ixx', ub_s_aff)

    # r/Iyy bounds
    lb_r = lb_theta.get_member('c')
    ub_r = ub_theta.get_member('c')
    lb_r_aff = np.zeros(4)
    ub_r_aff = np.zeros(4)
    for i in range(len(lb_r)):
        if lb_r[i] >= 0:
            lb_r_aff[i] = lb_r[i] / ub_theta.get_member('Iyy')
        else:
            lb_r_aff[i] = lb_r[i] / lb_theta.get_member('Iyy')
        if ub_r[i] >= 0:
            ub_r_aff[i] = ub_r[i] / lb_theta.get_member('Iyy')
        else:
            ub_r_aff[i] = ub_r[i] / ub_theta.get_member('Iyy')
    lb_theta_aff.set_member('c/Iyy', lb_r_aff)
    ub_theta_aff.set_member('c/Iyy', ub_r_aff)

    # b/Izz bounds
    lb_theta_aff.set_member('b/Izz', lb_theta.get_member('b') / ub_theta.get_member('Izz'))
    ub_theta_aff.set_member('b/Izz', ub_theta.get_member('b') / lb_theta.get_member('Izz'))

    # Ixx_cross bounds
    lb_Ixx_cross = lb_theta.get_member('Izz') - ub_theta.get_member('Iyy')
    if  lb_Ixx_cross >= 0:
        lb_theta_aff.set_member('Ixx_cross', lb_Ixx_cross / ub_theta.get_member('Ixx'))
    else:
        lb_theta_aff.set_member('Ixx_cross', lb_Ixx_cross / lb_theta.get_member('Ixx'))
    ub_Ixx_cross = ub_theta.get_member('Izz') - lb_theta.get_member('Iyy')
    if ub_Ixx_cross >= 0:
        ub_theta_aff.set_member('Ixx_cross', ub_Ixx_cross / lb_theta.get_member('Ixx'))
    else:
        ub_theta_aff.set_member('Ixx_cross', ub_Ixx_cross / ub_theta.get_member('Ixx'))

    # Iyy_cross bounds
    lb_Iyy_cross = lb_theta.get_member('Ixx') - ub_theta.get_member('Izz')
    if lb_Iyy_cross >= 0:
        lb_theta_aff.set_member('Iyy_cross', lb_Iyy_cross / ub_theta.get_member('Iyy'))
    else:
        lb_theta_aff.set_member('Iyy_cross', lb_Iyy_cross / lb_theta.get_member('Iyy'))
    ub_Iyy_cross = ub_theta.get_member('Ixx') - lb_theta.get_member('Izz')
    if ub_Iyy_cross >= 0:
        ub_theta_aff.set_member('Iyy_cross', ub_Iyy_cross / lb_theta.get_member('Iyy'))
    else:
        ub_theta_aff.set_member('Iyy_cross', ub_Iyy_cross / ub_theta.get_member('Iyy'))

    # Izz_cross bounds
    lb_Izz_cross = lb_theta.get_member('Iyy') - ub_theta.get_member('Ixx')
    if lb_Izz_cross >= 0:
        lb_theta_aff.set_member('Izz_cross', lb_Izz_cross / ub_theta.get_member('Izz'))
    else:
        lb_theta_aff.set_member('Izz_cross', lb_Izz_cross / lb_theta.get_member('Izz'))
    ub_Izz_cross = ub_theta.get_member('Iyy') - lb_theta.get_member('Ixx')
    if ub_Iyy_cross >= 0:
        ub_theta_aff.set_member('Izz_cross', ub_Izz_cross / lb_theta.get_member('Izz'))
    else:
        ub_theta_aff.set_member('Izz_cross', ub_Izz_cross / ub_theta.get_member('Izz'))

    return lb_theta_aff, ub_theta_aff
