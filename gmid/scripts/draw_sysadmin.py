"""
# influence diagrams for sys admin problem with a ring topology.
[SysAdmin](https://jair.org/index.php/jair/article/view/10341/24723)
Carlos Guestrin, et.al, "efficient solution algorithms for factored mdps", JAIR 2003

* draw sysadmin-ns_nd_ts_th
  * influence diagrams
  * primal graph
  * join tree
  * mini bucket tree
  * join graph
  * factor graph?
"""
PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
IMPORT_PYGRAPHVIZ=True
from gmid.fileio import *
from gmid.graph_algorithms import *
from gmid.graphical_models import *
from gmid.cte import *


def draw_influence_diagram(problem_name):
    file_name = os.path.join(BETA_PATH, problem_name)
    file_info = read_uai_id(file_name, False)

    # influence diagram
    sysadmin_id_obj = NxInfluenceDiagram()
    nvar = file_info["nvar"]
    var_types = file_info["var_types"]
    scopes = [[v.label for v in sc] for sc in file_info["scopes"]]
    scope_types = file_info["scope_types"]
    partial_elim_order = file_info["blocks"]
    sysadmin_id_obj.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
    sysadmin_id_nxgraph = sysadmin_id_obj.id
    sysadmin_id_obj.draw_diagram(file_name)

    # moralized primal graph
    moralized_id = moralize_dag(sysadmin_id_nxgraph)
    draw_nx_graph(file_name + "_moralized", moralized_id, [4, 9, 14])

    # join tree / mini-bucket tree -- schematic cte algorithm
    ordering_path = file_name + ".vo"
    factors = file_info['factors']
    blocks = file_info['blocks']
    fun_types = file_info["factor_types"]
    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg_obj = PrimalGraph(gm)
    draw_nx_graph(file_name + "_primal", pg_obj.nx_diagram, [4, 9, 14])
    try:
        ordering, iw = read_vo(ordering_path)
        iw = get_induced_width_from_ordering(pg_obj, ordering)
    except:
        ordering, iw = iterative_greedy_variable_order(2000, pg_obj.nx_diagram, ps=16, pe=-1, ct=inf, pv=blocks)
        write_vo_from_elim_order(ordering_path, ordering, iw)

    ibound = 4     # this will give join tree
    mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, random_partition=True)
    nx_mbt = mbtd.region_graph
    draw_nx_graph(file_name + "_mbt_i" + str(ibound), nx_mbt, [(4,0), (9,0), (14,0)])

    jgd = join_graph(mbtd, mini_buckets, ordering, False, False)
    nx_jgd =jgd.region_graph
    draw_nx_graph(file_name + "_jgd_i" + str(ibound), nx_jgd)


if __name__ == "__main__":
    draw_influence_diagram("sysadmin-4_1_2_4")



