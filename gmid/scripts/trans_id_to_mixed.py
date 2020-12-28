"""
input: influence diagram with *.uai, *.pvo, *.id
output: mixed mmap with *.uai, *.pvo, *.mi
"""
PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
import os


def run(problem_path):
    convert_uai_id_to_mixed(problem_path, problem_path + ".mixed")


if __name__ == "__main__":
    files = [f for f in os.listdir(TRANS_PATH)
              if os.path.isfile(os.path.join(TRANS_PATH, f))
              # and f.startswith("pomdp")
              and f.endswith(".uai")
              and not f.endswith(".mmap.uai")
              and not f.endswith(".mixed.uai")]
    for uai_name in files:
        run(os.path.join(TRANS_PATH, uai_name.replace(".uai", "")))

