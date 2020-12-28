PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.weighted_mbe4 import *
from gmid.search import message_graph_as_heuristic
import datetime

def run(name="pomdp5-6_4_3_5_3", ibound=5, tl=300, it=20, optimize_weight=1, optimize_cost=1, pseudo_tree= []):
# def run(name="car", pseudo_tree=[-1, 5, 5, 0, 3, 4], ibound=2, tl=3600, it=1, optimize_weight=0, optimize_cost=0):
    print("Start JOB:{},{} at {}".format(name, ibound, datetime.datetime.now()))
    problem_path = os.path.join(BETA_PATH, name)
    print("read problem files:{}".format(problem_path))
    file_info = read_uai_id(problem_path)
    if not pseudo_tree:
        pseudo_tree = file_info['pseudo_tree']
    assert pseudo_tree, "cannot export table without pseudo tree information"
    print("pseudo_tree:{}".format(pseudo_tree))
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
    # print("vars from gm class:{}".format(gm.variables))
    pg = PrimalGraph(gm)
    ordering, iw = read_vo(ordering_path)
    print("read ordering_path file:{}".format(ordering_path))
    iw = get_induced_width_from_ordering(pg, ordering)
    print('constrained induced width of the primal graph:{}'.format(iw))
    print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))

    #### mini bucket tree decomposition
    mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, ignore_msg=False, random_partition=False)
    jgd = join_graph(mbtd, mini_buckets, ordering, connect_mb_only=True, make_copy=False)
    add_const_factors(jgd, gm.variables, is_valuation, is_log)
    add_mg_attr_to_nodes(jgd)
    # add_mg_attr_to_edges(jgd, gm.variables, is_log, is_valuation)

    #### WMB heuristic
    verbose_level = 0
    wmb_str = "_".join( [name, "WeightedMBE", "iw="+str(iw), "ibd="+str(ibound),
                         "il="+str(it), "tl="+str(tl), "optw="+str(optimize_weight),
                         "optc="+str(optimize_cost)] )
    log_file_name = os.path.join(LOG_PATH_ID, wmb_str)
    wmb_heur = WeightedMBE4(verbose_level, jgd, ordering, weights, is_log, WEPS, log_file_name, ibound, mini_buckets, gm.variables)
    wmb_heur.print_log("WeightedMBE4")
    wmb_heur.print_log("START")
    wmb_heur.print_log("name:{}".format(name))
    wmb_heur.print_log("num vars:{}".format(file_info['nvar']))
    wmb_heur.print_log("num factors:{}".format(len(file_info['factors'])))
    wmb_heur.print_log("max domain:{}".format(max(file_info['domains'])))
    wmb_heur.print_log("max scope:{}".format(max([len(s) for s in file_info['scopes']])))
    wmb_heur.print_log('induced width:{}'.format(iw))
    wmb_heur.print_log("ordering:{}".format(ordering))
    wmb_heur.print_log("weights={}".format(weights))
    wmb_heur.print_log("connected components:{}".format(nx.number_connected_components(pg.G)))

    t0 = time.time()
    wmb_heur.schedule()
    wmb_heur.init_propagate()
    best_eu, final_Z = wmb_heur.propagate(time_limit=tl, iter_limit=it, optimize_weight=optimize_weight, optimize_cost=optimize_cost)
    t1 = time.time()
    wmb_heur.print_log("final Z:{}".format(final_Z))
    wmb_heur.print_log("final MEU:{}".format(best_eu))
    wmb_heur.print_log("Finish JOB:{}, with i-bound {} at {}".format(name, ibound, datetime.datetime.now()))
    wmb_heur.print_log("total time (sec) for solving problem:{}".format(t1-t0))
    wmb_heur.print_log("END")

    heur_info = message_graph_as_heuristic(wmb_heur.mg.message_graph, ordering, pseudo_tree,
                                       mini_buckets, num_var, num_fun)
    # print("bucket{}:{}".format(1, sorted(heur_info['bucket_msg'][1])))
    # print("bucket{}:{}".format(2, sorted(heur_info['bucket_msg'][2])))
    # print("bucket{}:{}".format(5, sorted(heur_info['bucket_msg'][5])))
    # print("bucket{}:{}".format(4, sorted(heur_info['bucket_msg'][4])))
    # print("bucket{}:{}".format(3, sorted(heur_info['bucket_msg'][3])))
    # print("bucket{}:{}".format(0, sorted(heur_info['bucket_msg'][0])))
    heur_name = "_".join([name, "i="+str(ibound), "it="+str(it)])
    write_mini_bucket_heuristic_from_info(os.path.join(BETA_PATH, heur_name), heur_info)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        name=str(sys.argv[1])
        name=name.split("/")[-1]
        print("read {}".format(name))
        name=name.replace(".uai", "")
        ibound = int(sys.argv[2])
        tl = int(sys.argv[3])
        it = int(sys.argv[4])
        optimize_weight = int(sys.argv[5])
        optimize_cost = int(sys.argv[6])
        run(name, ibound, tl, it, optimize_weight, optimize_cost)
    else:
        run()




