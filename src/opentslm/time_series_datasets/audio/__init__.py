# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-FileContributor: Fan Wu (ETH Zurich)
#
# SPDX-License-Identifier: MIT

"""Audio dataset package."""

from opentslm.time_series_datasets.audio.CareSoundDataset import CareSoundDataset
from opentslm.time_series_datasets.audio.CareSoundCoTDataset import CareSoundCoTDataset

__all__ = ["CareSoundDataset", "CareSoundCoTDataset"]
