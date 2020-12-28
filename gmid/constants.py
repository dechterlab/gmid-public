import sys
import os
import numpy as np

####
PRJ_PATH = "/home/junkyul/conda/gmid"
sys.path.append(PRJ_PATH)
LIB_PATH = os.path.join(PRJ_PATH, "libs")
sys.path.append(LIB_PATH)
PYGM_PATH = os.path.join(LIB_PATH, "pyGM")
sys.path.append(PYGM_PATH)

SRC_PATH = os.path.join(PRJ_PATH, "gmid")
PROBLEM_PATH = os.path.join(SRC_PATH, "problems")
PROBLEM_PATH_ID = os.path.join(SRC_PATH, "problems_id")
PROBLEM_PATH_MIXED = os.path.join(SRC_PATH, "problems_mixed")
PROBLEM_PATH_RELAXED = os.path.join(SRC_PATH, "problems_relaxed")
PRECOG_PATH = os.path.join(SRC_PATH, "problems_precog")
BETA_PATH = os.path.join(SRC_PATH, "beta")
TEST_PATH = os.path.join(SRC_PATH, "problems_test")
TRANS_PATH = os.path.join(SRC_PATH, "problems_trans")

LOG_PATH = os.path.join(SRC_PATH, "logs")
LOG_PATH_ID = os.path.join(SRC_PATH, "logs_id")
LOG_PATH_MIXED = os.path.join(SRC_PATH, "logs_mixed")
LOG_PATH_RELAXED = os.path.join(SRC_PATH, "logs_relaxed")

from libs import pyGM
from sortedcontainers import SortedSet  # installed in site-package
import networkx as nx   # how to localize this?
import numpy as np

####
debug = False
orderMethod = 'F'  # used for factor class in pyGM
ZERO = 0
TOL = 1e-8
ONE = np.float64(1.0)
np.seterr(all='ignore') # ignore warn raise
# EXPLIM = 709.78271289338397     # np.log(1.7976931348623157e+308), numpy.finfo(numpy.float64).max
# NPEPS = 1.49e-08    # step Step size used for the finite difference approximation. It defaults to sqrt(numpy.finfo(float).eps), which is approximately 1.49e-08
# GRADEPS = 1e-6
WEPS = 1e-6    # w for max or sum when evaluating bound (max/sum) or pseudomarginal (max)
WINF = 1e+6    # np.finfo('d').max  # 1.7976931348623157e+308
# WZERO = 1e-6    # set to zero below this weight


IMPORT_PYGRAPHVIZ=False
