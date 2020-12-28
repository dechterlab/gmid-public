PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
import os

def run(problem_path):
    convert_erg_to_uai(problem_path + ".erg", problem_path)


if __name__ == "__main__":
    file = [f for f in os.listdir(TRANS_PATH) if os.path.isfile(os.path.join(TRANS_PATH, f)) and f.endswith(".erg")]
    for uai_name in file:
        run(os.path.join(TRANS_PATH, uai_name.replace(".erg", "")))
