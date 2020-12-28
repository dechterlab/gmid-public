PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
import os


convert_uai_to_limid(os.path.join(PROBLEM_PATH_ID, "mdp1-4_2_2_5"), os.path.join(TRANS_PATH, "mdp1-4_2_2_5"))
