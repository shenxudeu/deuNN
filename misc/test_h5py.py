import numpy as np
import os
import tempfile

from pylearn2.testing.skip import skip_if_no_h5py
from pylearn2.testing.datasets import (
        random_dense_design_matrix,
        random_one_hot_dense_design_matrix,
        random_one_hot_topological_dense_design_matrix)


