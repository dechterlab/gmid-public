"""
input: mmap with *.uai, *.map
output: mmap with *.uai, *.pvo, *.mi
"""

# load mmap by read_uai_mmap() --> file_info
# then, write uai_mixed(file_name, file_info)
PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
import os


def run(problem_path):
    convert_mmap_to_mixed(problem_path, problem_path+".mixed")


if __name__ == "__main__":
    files = [f for f in os.listdir(TRANS_PATH) if os.path.isfile(os.path.join(TRANS_PATH, f))
              # and f.startswith("mdp4")
              and f.endswith(".mmap.uai")]
    for uai_name in files:
        run(os.path.join(TRANS_PATH, uai_name.replace(".uai", "")))
