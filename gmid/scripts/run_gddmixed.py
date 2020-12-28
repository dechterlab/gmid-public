PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.gddmixed import *


def run(name="ID_from_BN_0_w28d6.mixed", ibound=1, time_limit=3600, iter_limit=1000):
# def run(name="car.mixed", ibound=10, time_limit=3600, iter_limit=1000):
    problem_path = os.path.join(TEST_PATH, name)
    ordering_path = problem_path + ".vo"
    file_info = read_uai_mixed(problem_path)
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']

    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = True
    is_valuation = False
    gm = GraphicalModel(factors, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = read_vo(ordering_path)
    print("read ordering_path file:{}".format(ordering_path))
    iw = get_induced_width_from_ordering(pg, ordering)
    print('constrained induced width of the primal graph:{}'.format(iw))
    print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))

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
    gdd_mp.print_log("GDDMixed")
    gdd_mp.print_log("gd_options:{}".format(gd_options))
    gdd_mp.print_log("weight_options:{}".format(gd_options))
    gdd_mp.print_log("start solving:{}".format(problem_path))
    gdd_mp.print_log("iw={}, elim_order={}, ibound={}".format(iw, ordering, ibound))
    gdd_mp.print_log("nv={}, nf={}".format(file_info['nvar'], len(file_info['factors'])))
    gdd_mp.print_log("weights={}".format(weights))
    gdd_mp.print_log("number of clusters={}".format(len(jgd.region_graph.nodes())))

    #### Start GDD
    gdd_mp.schedule()
    gdd_mp.init_propagate()
    best_mmap = gdd_mp.propagate(time_limit=time_limit, iter_limit=iter_limit, cost_options=gd_options, weight_options=gd_options)
    gdd_mp.print_log("finished solving:{}".format(name))
    gdd_mp.print_log("best bound found={}".format(best_mmap))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name=str(sys.argv[1])
        name=name.split("/")[-1]
        if not name.endswith(".mixed.uai"):
            assert False, name + " is wrong input file"
        name = name.replace(".uai", "")
        ibound = int(sys.argv[2])
        time_limit = int(sys.argv[3])
        iter_limit = int(sys.argv[4])
        run(name, ibound, time_limit, iter_limit)
    run()
