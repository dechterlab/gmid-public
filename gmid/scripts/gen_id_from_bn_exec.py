PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gen_id_from_bn import generate_id_from_bn
from run_cte import run_cte


### create a file -> read and solve it by CTE
def run():
    name = "BN_14"
    generate_id_from_bn(name + ".uai", 0.1)

    print("solve")
    name = "ID_from_" + name
    run_cte(name=name, format_type="uai")
    # run_cte(name=name, format_type="erg")
    # run_cte(name=name, format_type="limid")


if __name__ == "__main__":
    run()
