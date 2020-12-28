PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.cte import *
import pprint
import datetime
pp = pprint.PrettyPrinter(indent=4)


def run_cte(name="mdp1-4_2_2_5", ibound=1):
    print("Start JOB:{},{} at {}".format(name, ibound, datetime.datetime.now()))
    problem_path = os.path.join(PROBLEM_PATH_RELAXED, name)
    print("read problem files:{}".format(problem_path))
    file_info = read_uai_id(problem_path)
    ordering_path = problem_path + '.vo'
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    num_var = file_info['nvar']
    num_fun = len(file_info['factors'])

    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    print("read ordering_path file:{}".format(ordering_path))
    ordering, iw = read_vo(problem_path + '.vo')
    iw = get_induced_width_from_ordering(pg, ordering)
    print('constrained induced width of the primal graph:{}'.format(iw))
    print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))


    #### mini bucket decomposition and join graph decomposition
    mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=ordering, ibound=ibound, ignore_msg=False, random_partition=True)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)

    verbose_level = 0
    mbe_str = "_".join([name, "RelaxedMBE", "iw="+str(iw), "ibd="+str(ibound)])
    log_file_name = os.path.join(LOG_PATH_ID, mbe_str)
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=log_file_name)
    tree_mp.print_log("RelaxedMBE")
    tree_mp.print_log("START")
    tree_mp.print_log("name:{}".format(name))
    tree_mp.print_log("num vars:{}".format(file_info['nvar']))
    tree_mp.print_log("num factors:{}".format(len(file_info['factors'])))
    tree_mp.print_log("max domain:{}".format(max(file_info['domains'])))
    tree_mp.print_log("max scope:{}".format(max([len(s) for s in file_info['scopes']])))
    tree_mp.print_log('induced width:{}'.format(iw))
    tree_mp.print_log("ordering:{}".format(ordering))
    tree_mp.print_log("connected components:{}".format(nx.number_connected_components(pg.G)))

    t0 = time.time()
    tree_mp.schedule()
    tree_mp.init_propagate()
    bound_at_root, Z_at_root = tree_mp.propagate(propagation_type='ve', max_sum=True)
    t1 = time.time()
    tree_mp.print_log("final Z:{}".format(Z_at_root))
    tree_mp.print_log("final MEU:{}".format(bound_at_root))
    tree_mp.print_log("Finish JOB:{}, with i-bound {} at {}".format(name, ibound, datetime.datetime.now()))
    tree_mp.print_log("total time (sec) for solving problem:{}".format(t1-t0))
    tree_mp.print_log("END")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        name=str(sys.argv[1])
        name=name.split("/")[-1]
        print("read {}".format(name))
        # if not name.endswith(".uai"):
        #     assert False, name + " is wrong input file"
        name=name.replace(".uai", "")
        ibound = int(sys.argv[2])
        run_cte(name, ibound)
    else:
        run_cte()
