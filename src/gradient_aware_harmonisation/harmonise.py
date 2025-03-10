import numpy as np
import tensorflow as tf

from scipy.interpolate import make_interp_spline
from typing import Tuple, List, Union, Callable, Optional, Any
from gradient_aware_harmonisation.typings import ResDict, ResAdjustDict

LFlInt = List[Union[int, float]]


class Harmonise:
    def __call__(
        self,
        x: Tuple[LFlInt, LFlInt],
        y: Tuple[LFlInt, LFlInt],
        t0: Union[float, int],
        smooth: float = 1.0,
        t_converge: Optional[Union[int, float]] = None,
        **kwargs
    ) -> ResDict:
        # compute functions
        f1 = make_interp_spline(x[0], y[0], **kwargs)
        f2 = make_interp_spline(x[1], y[1], **kwargs)

        # compute derivatives
        df1 = f1.derivative()
        df2 = f2.derivative()

        # adjust first-order func (slope-only)
        match_deriv = self.adjust_func(df1, df2, x[1], t0, inverse=True, **kwargs)

        # adjust zero-order func (abs-only)
        match_abs = self.adjust_func(f1, f2, x[1], t0, **kwargs)

        # adjust zero-order of adjusted first-order func (full adj)
        match = self.adjust_func(f1, match_deriv["f2"], x[1], t0, **kwargs)

        # interpolate between full adj and zero-order adjusted func
        f2_interp = self.f2_interpolate(
            x[1], match_abs["f2"], match["f2"], smooth=smooth, t_converge=t_converge, **kwargs
        )
        df2_interp = f2_interp.derivative()

        # truncate functions
        res: ResDict = self.truncate_func(
            [f1, f2, match["f2"], match_abs["f2"], f2_interp],
            [df1, df2, match["df2"], match_abs["df2"], df2_interp],
            x,
            t0,
        )

        return res


    def adjust_func(
        self,
        f1: Callable[[Union[int, float]], Union[int, float]],
        f2: Callable[[Union[int, float, LFlInt]], Union[int, float]],
        x2: LFlInt,
        t0: Union[float, int],
        inverse: bool = False,
        **kwargs
    ) -> ResAdjustDict:
        diff = f1(t0)-f2(t0)
        y2_match = f2(x2)+diff
        if inverse:
            df2_match = make_interp_spline(x2, y2_match, **kwargs)
            f2_match = df2_match.antiderivative()
        else:
            f2_match = make_interp_spline(x2, y2_match, **kwargs)
            df2_match = f2_match.derivative()

        res: ResAdjustDict = dict(f2=f2_match, df2=df2_match)

        return res


    # interpolate between zero-order and first-order match
    def f2_interpolate(
        self,
        x2: LFlInt,
        f2: Callable[[LFlInt], LFlInt],
        f2_match: Callable[[LFlInt], LFlInt],
        smooth: float,
        t_converge: Optional[Union[int, float]],
        **kwargs
    ) -> Callable[[LFlInt], Callable]:
        if t_converge is None:
            decay_end = len(f2(x2))
        else:
            decay_end = self.idx_x(x2, t_converge)

        # decay function
        func_decay = tf.keras.optimizers.schedules.CosineDecay(1.0, decay_end)
        # compute weight
        k_seq = np.stack([smooth * func_decay(x) for x in range(len(f2(x2)))])
        # compute adjusted observations
        y_new = np.stack(
            [
                np.mean(np.add(k * f2_match(x2)[i], (1 - k) * f2(x2)[i]))
                for i, k in enumerate(k_seq)
            ]
        )
        # estimate function
        f_interpol = make_interp_spline(x2, y_new, **kwargs)
        return f_interpol


    def truncate_func(
        self,
        f: List[Any],
        df: List[Any],
        x: Tuple[LFlInt, LFlInt],
        t0: Union[int, float],
    ) -> ResDict:
        i_x1 = self.idx_x(x[0], t0)
        i_x2 = self.idx_x(x[1], t0)

        if i_x1 > i_x2:
            res: ResDict = dict(
                f1=f[0](x[0])[:int(i_x1+1)],
                df1=df[0](x[0])[:int(i_x1+1)],
                f2=f[1](x[1])[i_x2:],
                df2=df[1](x[1])[i_x2:],
                f2_adj=f[2](x[1])[i_x2:],
                df2_adj=df[2](x[1])[i_x2:],
                f2_abs=f[3](x[1])[i_x2:],
                df2_abs=df[3](x[1])[i_x2:],
                f2_intpol=f[4](x[1])[i_x2:],
                df2_intpol=df[4](x[1])[i_x2:],
                x1=x[0][:int(i_x1+1)],
                x2=x[1][i_x2:],
            )
        else:
            res: ResDict = dict(
                f1=f[0](x[0])[i_x1:],
                df1=df[0](x[0])[i_x1:],
                f2=f[1](x[1])[:int(i_x2+1)],
                df2=df[1](x[1])[:int(i_x2+1)],
                f2_adj=f[2](x[1])[:int(i_x2+1)],
                df2_adj=df[2](x[1])[:int(i_x2+1)],
                f2_abs=f[3](x[1])[:int(i_x2+1)],
                df2_abs=df[3](x[1])[:int(i_x2+1)],
                f2_intpol=f[4](x[1])[:int(i_x2+1)],
                df2_intpol=df[4](x[1])[:int(i_x2+1)],
                x1=x[0][i_x1:],
                x2=x[1][:int(i_x2+1)],
            )
        return res


    def idx_x(self, x: List[Union[int, float]], t0: Union[float, int]) -> int:
        for i in range(len(x)-1):
            check = False
            if x[i] <= t0 and x[i + 1] >= t0:
                check = True
                idx0 = i+1
                break
        if check is not True:
            idx0 = None
            raise ValueError(
                f"The provided target t0={t0} is not covered by both provided timeseries."
            )
        return idx0


harmonise = Harmonise()
