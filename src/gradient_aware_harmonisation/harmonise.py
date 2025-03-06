import numpy as np

from typing import Tuple, List, Union
from gradient_aware_harmonisation.typings import ResDict


class Harmonise:
    def __call__(
        self,
        f1: Tuple[List[Union[int, float]], List[Union[int, float]]],
        f2: Tuple[List[Union[int, float]], List[Union[int, float]]],
        x0: float,
    ) -> ResDict:
        # compute correction in absolute values
        (correction_abs, case, idx) = self.test_abs_val(f1, f2, x0)

        # adjust y-value of f2 if necessary
        cor_f2 = self.corrected_abs_val_f2(correction_abs, f2)
        # cut timeseries outside range of interest
        new_f1, new_f2 = self.cut_timeseries(f1, cor_f2, idx[0], idx[1])
        # save results in dict
        res: ResDict = dict(f1=new_f1, f2=new_f2, case=case)
        return res

    # get index where f_x = x0
    # if x0 is not covered by f raise value error
    def idx_x(self, x: List[Union[int, float]], x0: float) -> int:
        for i in range(len(x) - 1):
            check = False
            if x[i] <= x0 and x[i + 1] > x0:
                check = True
                return i
        if check is not True:
            raise ValueError(
                f"The provided target x0={x0} is not covered by both provided timeseries."
            )

    # get x,y corresponding to target_x
    def get_target_xy(
        self, dat_xy: Tuple[List[Union[int, float]], List[Union[int, float]]], x0: float
    ) -> Tuple[Tuple[Union[float, int], Union[int, float]], int]:
        idx = self.idx_x(dat_xy[0], x0=x0)
        return (dat_xy[0][idx], dat_xy[1][idx]), idx

    # test whether absolute y-values are equal at x0?
    def test_abs_val(
        self,
        f1: Tuple[List[Union[float, int]], List[Union[float, int]]],
        f2: Tuple[List[Union[int, float]], List[Union[float, int]]],
        x0: float,
    ) -> Tuple[float, str, List[int]]:
        (tx1, ty1), idx_f1 = self.get_target_xy(f1, x0)
        (tx2, ty2), idx_f2 = self.get_target_xy(f2, x0)
        if ty1 < ty2:
            case = "f1(x0) < f2(x0)"
            abs_val = ty1 - ty2
        elif ty1 > ty2:
            case = "f1(x0) > f2(x0)"
            abs_val = ty1 - ty2
        elif ty1 == ty2:
            case = "f1(x0) == f2(x0)"
            abs_val = 0
        return abs_val, case, [idx_f1, idx_f2]

    # correct y-value such that f1 and f2 match at x0
    def corrected_abs_val_f2(
        self,
        correction_abs: float,
        f2: Tuple[List[Union[float, int]], List[Union[float, int]]],
    ) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
        corrected_x_f2 = np.add(f2[1], correction_abs)
        return f2[0], list(np.squeeze(corrected_x_f2))

    # cut timeseries
    def cut_timeseries(
        self,
        f1: Tuple[List[Union[float, int]], List[Union[float, int]]],
        f2: Tuple[List[Union[float, int]], List[Union[float, int]]],
        idx_f1: int,
        idx_f2: int,
    ) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
        if idx_f1 >= idx_f2:
            f1_new = (f1[0][:idx_f1], f1[1][:idx_f1])
            f2_new = (f2[0][idx_f2:], f2[1][idx_f2:])
        else:
            f1_new = (f1[0][idx_f1:], f1[1][idx_f1:])
            f2_new = (f2[0][:idx_f2], f2[1][:idx_f2])
        return f1_new, f2_new


harmonise = Harmonise()
