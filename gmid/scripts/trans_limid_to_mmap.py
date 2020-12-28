"""
input:  influence diagarm in LIMID format, *.limid
output: marginal MAP in uai format, *.mmap.uai, *.mmap.map
"""
PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from libs.pyGM.filetypes import *
import os

# TODO schematic translation -> only shows problem stats after trans


def run(problem_path):
    convert_uai_to_limid(problem_path, problem_path + ".limid")
    C,D,U = readLimidCRA(problem_path + ".limid")
    F,Q = LimidCRA2MMAP( C,D,U )
    writeUai(problem_path +'.mmap.uai', F )
    writeOrder(problem_path +'.mmap.map', Q )
    os.remove(problem_path + ".limid")


if __name__ == "__main__":
    files = [f for f in os.listdir(TRANS_PATH) if os.path.isfile(os.path.join(TRANS_PATH, f))
              and f.startswith("pomdp")
              and f.endswith(".uai")
              and not f.endswith(".mixed.uai")
              and not f.endswith(".mmap.uai")]
    for uai_name in files:
        run(os.path.join(TRANS_PATH, uai_name.replace(".uai", "")))
