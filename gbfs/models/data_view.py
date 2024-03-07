from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DataCollection:
    x: pd.DataFrame
    y: pd.DataFrame
    x_y: pd.DataFrame


@dataclass
class DataProps:
    n_labels: int
    labels: np.ndarray

    n_features: int
    features: pd.DataFrame
    feature_costs: Dict[str, float]


@dataclass
class DataView:
    data_props: DataProps
    data: DataCollection
    norm_data: DataCollection


@dataclass
class FeaturesGraph:
    sep_matrix: np.ndarray
    reduced_sep_matrix: np.ndarray
