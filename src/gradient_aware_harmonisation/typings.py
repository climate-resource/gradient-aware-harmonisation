from typing import TypedDict, Tuple, List, Union

class ResDict(TypedDict):
    f1: Tuple[List[Union[int,float]], List[Union[int,float]]]
    f2: Tuple[List[Union[int,float]], List[Union[int,float]]]
    case: str