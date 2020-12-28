PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.cte import *
import pprint
pp = pprint.PrettyPrinter(indent=4)


def run(problem_path):
    file_info = read_limid(problem_path)
    name=problem_path.split("/")[-1]
    name=name.replace(".limid","")
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']

    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = iterative_greedy_variable_order(1000, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)

    mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=ordering, ibound=iw, ignore_msg=False, random_partition=False)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)
    verbose_level = 0
    log_file_name = os.path.join(LOG_PATH, name)
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=log_file_name)
    tree_mp.schedule()
    tree_mp.init_propagate()
    bound_at_root, Z_at_root = tree_mp.propagate()
    marginals = tree_mp.bounds(root_only=False)
    clusters = sorted(tree_mp.schedule_graph.nodes(), key=lambda x: tree_mp.elim_order.index(x[0]))
    tree_mp.print_log("START")
    tree_mp.print_log("name:{}".format(name))
    tree_mp.print_log("max domain size:{}".format(max(file_info['domains'])))
    tree_mp.print_log("num factors:{}".format(len(file_info['factors'])))
    tree_mp.print_log("num vars:{}".format(file_info['nvar']))
    tree_mp.print_log("max scope size:{}".format(max([len(s) for s in file_info['scopes']])))
    tree_mp.print_log('induced width:{}'.format(iw))
    tree_mp.print_log("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.G)))
    tree_mp.print_log("final MEU:{}".format(bound_at_root))
    tree_mp.print_log("final Z:{}".format(Z_at_root))
    tree_mp.print_log("END")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(str(sys.argv[1]))
    else:
        for f in os.listdir(PRECOG_PATH):
            run(os.path.join(PRECOG_PATH, f))
