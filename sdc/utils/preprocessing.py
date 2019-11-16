from typing import Tuple
from collections import Counter

import numpy as np


class FixedSlidingWindow(object):
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

            Raises:
                AssertionError: an error occur when argument overlap_rate under 0.0 or over 1.0.n error occurred.

            """
    def __init__(self, window_size: int, overlap_rate: float, step_size: int = None) -> None:
        self.window_size = window_size

        if overlap_rate is None and step_size is not None:
            if 0 < step_size:
                self.overlap = int(step_size)
        else:
            if not 0.0 < overlap_rate <= 1.0:
                raise AssertionError("overlap_rate ranges from 0.0 to 1.0")

            self.overlap = int(window_size * overlap_rate)

    def transform(self, x: np.ndarray) -> np.array:
        """

        Args:
            x: 2 or 3 dim of np.ndarray

        Returns:
            np.ndarray
        """
        seq_len = x.shape[0]
        assert seq_len > self.window_size
        data = [x[i:i + self.window_size] for i in range(0, seq_len - self.window_size, self.overlap)]

        data = np.stack(data, 0)
        return data

    @staticmethod
    def clean(labels: np.ndarray) -> np.array:
        tmp = []
        for l in labels:
            window_size = len(l)
            c = Counter(l)
            common = c.most_common()
            values = list(c.values())
            if common[0][0] == 0 and values[0] == window_size // 2:
                label = common[1][0]
            else:
                label = common[0][0]

            tmp.append(label)

        return np.array(tmp)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.array, np.array]:
        data = self.transform(x)
        label = self.transform(y)
        label = self.clean(label)
        return data, label
