PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.gddmixed import *


def run(name="car.mixed", ibound=10, time_limit=100, iter_limit=50):
    problem_path = os.path.join(TEST_PATH, name)
    file_info = read_uai_mixed(problem_path)
    factors = file_info['factors']
    weights = [1.0] * file_info['nvar']
    is_log = True
    is_valuation = False
    gm = GraphicalModel(factors, weights, is_log=is_log)
    ordering = [6, 1, 2, 5, 4, 3, 0]

    #### join graph decomposition
    verbose_level = 0
    mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, random_partition=True)
    jgd = join_graph(mbtd, mini_buckets, ordering, connect_mb_only=False, make_copy=False)
    add_mg_attr_to_nodes(jgd)
    add_mg_attr_to_edges(jgd, gm.variables, is_log)
    add_const_factors(jgd, gm.variables, is_valuation, is_log)

    #### GDD for mixed mmap
    gdd_str = '_GddMixed_i=' + str(ibound) + '_init=' + str(False)
    log_file_name = os.path.join(LOG_PATH_MIXED, name + gdd_str)
    gd_options = {
        'gd_steps': 10, 'tol': TOL,
        'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5, 'ls_tol': TOL
    }
    assert is_log, "GDD for mixed mmap operates in log space"
    gdd_mp = GddMixed(verbose_level, jgd, ordering, weights, is_log, WEPS, log_file_name)

    #### Start GDD
    gdd_mp.schedule()
    gdd_mp.init_propagate()
    best_mmap = gdd_mp.propagate(time_limit=time_limit, iter_limit=iter_limit, cost_options=gd_options, weight_options=gd_options)
    gdd_mp.print_log("finished solving:{}".format(name))
    gdd_mp.print_log("best bound found={}".format(best_mmap))


if __name__ == "__main__":
    run()
