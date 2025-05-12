"""
Type hinting support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

NP_FLOAT_OR_INT: TypeAlias = Union[np.floating[Any], np.integer[Any]]
"""
Type alias for a numpy float or int (not complex)
"""

NP_ARRAY_OF_FLOAT_OR_INT: TypeAlias = npt.NDArray[NP_FLOAT_OR_INT]
"""
Type alias for an array of numpy float or int (not complex)
"""
