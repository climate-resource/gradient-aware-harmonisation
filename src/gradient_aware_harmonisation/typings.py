from typing import TypedDict, List, Union, Callable

LFlInt = List[Union[int, float]]

class ResDict(TypedDict):
    f1 = LFlInt,
    df1 = LFlInt,
    f2 = LFlInt,
    df2 = LFlInt,
    f2_adj = LFlInt,
    df2_adj = LFlInt,
    f2_abs = LFlInt,
    df2_abs = LFlInt,
    f2_intpol = LFlInt,
    df2_intpol = LFlInt,
    x1 = LFlInt,
    x2 = LFlInt


class ResAdjustDict(TypedDict):
    f2 = Callable,
    df2 = Callable