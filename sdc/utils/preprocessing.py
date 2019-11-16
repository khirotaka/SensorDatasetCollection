"""Preprocessing functions for Time Series."""

from typing import Tuple
from collections import Counter

import numpy as np


class FixedSlidingWindow:
    """Fixed sliding window.

    Examples::

        >>> import numpy as np
        >>> from sdc.utils.preprocessing import FixedSlidingWindow
        >>> x = np.random.randn(1024, 23)
        >>> y = np.random.randint(0, 9, 1024)
        >>> sw = FixedSlidingWindow(256, overlap_rate=0.5)
        >>> x, y = sw(x, y)
        >>> x.shape     # [6, 256, 23]
        >>> y.shape     # [6, ]

    Args:
        window_size: int
        overlap_rate: float
        step_size: int (default, None)

    Raises:
        AssertionError: an error occur when
            argument overlap_rate under 0.0 or over 1.0.n error occurred.

    """
    def __init__(self, window_size: int, overlap_rate: float, step_size=None) -> None:
        """
        Initializer of FixedSlidingWindow.

        Args:
            window_size: int
            overlap_rate: float
            step_size: int (default, None)

        """
        self.window_size = window_size

        if overlap_rate is None and step_size is not None:
            if step_size > 0:
                self.overlap = int(step_size)
        else:
            if not 0.0 < overlap_rate <= 1.0:
                raise AssertionError("overlap_rate ranges from 0.0 to 1.0")

            self.overlap = int(window_size * overlap_rate)

    def transform(self, inputs: np.ndarray) -> np.array:
        """

        Args:
            inputs: 2 or 3 dim of np.ndarray

        Returns:
            np.ndarray
        """
        seq_len = inputs.shape[0]
        if not seq_len > self.window_size:
            raise Exception("window size must be smaller then input sequence length.")

        data = [
            inputs[i:i+self.window_size] for i in range(0, seq_len-self.window_size, self.overlap)
        ]

        data = np.stack(data, 0)
        return data

    @staticmethod
    def clean(labels: np.ndarray) -> np.array:
        """
        Clean up

        Args:
            labels:

        Returns:

        """
        tmp = []
        for lbl in labels:
            window_size = len(lbl)
            counter = Counter(lbl)
            common = counter.most_common()
            values = list(counter.values())
            if common[0][0] == 0 and values[0] == window_size // 2:
                label = common[1][0]
            else:
                label = common[0][0]

            tmp.append(label)

        return np.array(tmp)

    def __call__(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.array, np.array]:
        data = self.transform(data)
        label = self.transform(target)
        label = self.clean(label)
        return data, label
