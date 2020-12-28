PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
import pprint
pp = pprint.PrettyPrinter(indent=4)


def run(name="ID_from_BN_0_w33d11", file_type="uai"):
    problem_path = os.path.join(PROBLEM_PATH_ID, name)
    ordering_path = problem_path + '.vo'

    read_func = {"uai" : read_uai_id, "mixed" : read_uai_mixed}

    file_info = read_func[file_type](problem_path, sort_scope=False, skip_table=True)
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']

    # print("create a graphical model")
    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    try:
        # print("read ordering_path file:{}".format(ordering_path))
        ordering, iw = read_vo(ordering_path)
        iw = get_induced_width_from_ordering(pg, ordering)
    except:
        print("find a variable ordering")
        ordering, iw = iterative_greedy_variable_order(2000, pg.nx_diagram, ps=16, pe=-1, ct=inf, pv=blocks)
        write_vo_from_elim_order(ordering_path, ordering, iw)
        print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))
        print('constrained induced width of the primal graph:{}'.format(iw))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name=str(sys.argv[1])
        name=name.split("/")[-1]
        name=name.replace(".uai", "")
        run(name, str(sys.argv[2]))
    else:
        run()
