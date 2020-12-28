# gmid
graphical model inference algorithms for influence diagrams.
* related papers
  * Junkyu Lee, Radu Marinescu, Alexander Ihler, and Rina Dechter. "A Weighted Mini-Bucket Bound for Solving Influence Diagrams" in Proceedings of UAI 2019.
  * Junkyu Lee, Alexander Ihler, and Rina Dechter. "Join Graph Decomposition Bounds for Influence Diagrams" in Proceedings of UAI 2018.

* algorithms
  * variable elimination with valuation algebra
  * mini-bucket elimination with valuation algebra
  * join graph decomposition bounds for id
* translations
  * influence diagram to marginal MAP
  * factored mdp/pomdp generator and translator to influence diagrams
* information relaxation by identifying minimum sufficient information set
* visualization of graphical models and related graphs 
  * influence diagram, primal graph, join tree, join graph, etc


# dependencies 
* python2
 [pyGM](https://github.com/ihler/pyGM)
  * put source under /gmid/lib/pyGM
* networkx 1.11
  * install from source [networkx](https://github.com/networkx/networkx/tree/v1.11)
  ```
  $python setup.py install --user
  $sudo python setpu.py install
  ```
* SortedContainers
  ```
  $pip install sortedcontainers
  ```
* graphviz and pygraphviz
  ```
  $sudo apt-get install graphviz libgraphviz-dev pkg-config
  $pip2 install pygraphviz
  ```
* numpy

* scipy 1.2.1

```
$ conda create -n gmid python=2.7
$ conda activate gmid
$ conda install -c conda-forge sortedcontainers
$ conda install scipy=1.2.1
$ conda install networkx=1.11
```

# orgainzations
* gmid
  * gmid
    * /logs: store log files
    * /problems: store influence diagram problem files
    * /scripts: scripts for running algorithms
  * lib
    * /pyGM: gmid uses classes related to factors and variables
  * data
    * /benchmark: set of problems used in uai 2018 paper
    * /executables: some linux executables for comparison
    * /log-process: draw plots from data
    * /uai2018-paper: data related to uai 2018 paper

# file formats
  * ID-UAI format, variant of uai format for influence diagram
    * *.uai defines factors
    * *.id defines identity of chance/decision variables and probability/utility functions
    * *.pvo or *.vo defines partial/total elimination ordering dictated by sequential decisions and observations
  * MI-UAI format, variant of uai format for marginal map
    * *.uai defines factors
    * *.mi defines decision and summation variables
    * *.pvo or *.vo defines partial/total elimination ordering
  * ID-ERG format, variant of erg format for influence diagram
    * single file defines factors, identity, and total elimination ordering
  * LIMID format, another variant of uai format called limid for influence diagram
    * this also defines all elements for influence diagrams in 1 file
  * fileio.py
    * provides reader/writer to each format
    * provides conversion between them
    * problems_id stores influence diagram in the first format that separates factors, identity, and constriants
  * convention
    * problems_id: collection of uai, id, pvo
    * problems_mixed: collection of uai, mi, pvo --> MMAP with mixed order
    * problems_mmap: collection of uai, and map files --> standard UAI MMAP format
    * problems_relaxed:  collection of uai, id, pvo but resulting from information relaxation by MSIS
    * problems_trans: stores converted files

# algorithms
* overall process
  1. read a graphical model problem and create list of factors and variables (valuations)
  2. create a graphical model object with factors and variables
  3. create a primal graph object from graphical model object
  4. find a constrained variable elimination ordering with a primal graph object
  5. create a mini_bucket_tree with an i-bound parameter by mini_bucket_tree(graphical_model, ordering, ibound)
  6. solve problem by CTE(clutster tree elim algorithm) --> exact solver or mini-bucket elimination stops here
  7. create a join graph from mini_bucket_tree by join_graph(mini_bucket_tree.message_graph, mini_buckets, ordering)
  8. add functions, scopes, and various algorithm specific information to message graphs
  9. solve problem by GddIdHingeShiftProjected(JGDID algorithm)


* graph_algorithms.py
  * visuzlize NxDiagram, PrimalGraph, InfluenceDiagram, Limid...
    - ```python 
      class NxDiagram()
      class PrimalGraph()
      class NxInfluenceDiagram()
      class NxFactoredMDP() // added to visualize multi decision variables per stage
      ```
  * algorithms for finding minimum separating information set in DAG and testing d-separation
    - ```python 
      def get_relaxed_influence_diagram()
      ```
  * algorithms for finding variable elimination ordering
    - ```python 
      def iterative_greedy_variable_order()
      ```
  * Mini Bucket Tree and Join graph decomposition to be used as region grphas
    - ```python 
      def mini_bucket_tree()
      def join_graph()
      def gdd_graph()
      ```
  * Message Graph class
    - ```python
      class MessageGraph()
      ```
    - create a MessageGraph object with a region_graph and elimination order
    - region graph can be bucket tree, mini-bucket tree, join graph structured by mini bucket tree, or gdd graph
  * additional functions defining additional node/ edge attributes for region graphs and message graphs


* graphical_models.py
  * ```python
    class GraphicalModel()
    ```
  * graphcial model class is a container for variables, factors, elimination order, etc.
  * create a graphical model with a list of factors, a list of weights of variables and elimination order

* message_passing.py
  * ```python
    class MessagePassing()
    ```
  * MessagePassing class is abstract class for message passing algorithms
  * MessagePassing object holds a MessageGraph, list of weights, elimination order, and path to the log file.
  * MessagePassing algorithms implement 
    - ```python
      def schedule()
      def init_propagate()
      def propagate()
      def _propagate_one_pass()
      def bounds()
      ```
* cte.py implements cluster tree elimination (variable elimination, mini-bucket elimination)
  * ```python
    class CTE(MessagePassing):
    ```
  * use common interface defined by MessagePassing

* gdd_hinge_shift_prj.py implements JGDID algorithm
  * ```python
    class GddIdHingeShiftProjected(MessagePassing):
    ```
  * use common interface defined by MessagePassing

* gddmixed.py implements GDD algorithm for MMAP
  * ```python
    class GddMixed(MessagePassing):
    ```
  * use common interface defined by MessagePassing

* valuation.py
  * ```python
    class Valuation():
    ```
  * Valuation class is a wrapper class of factor class in pyGM that supports valuation algebra

* constats.py
  * defines various global variables
  * PRJ_PATH = "/home/junkyul/gmid" should be replaced by proper path (same for all running examples)

* run_weightedmbe.py
  * ```
    class WeightedMBE(MessagePassing):
    ```
  * implementing mbe with valuation algebra + optimizing dynamic gdd bound per layer of mini-buckets


# Script examples 

* running algorithms
  * run_vo.py
    * given ID-UAI or MI-UAI (uai, id/mi, pvo), find vo (total elim order)
    * generating vo file from pvo file can be done before running each algorithm
  
  * run_cte.py
    * run CTE algorithm for influence diagram
    * input files are stored in problems_id

  * run_gddhingeshifprj.py/ run_gddhingeshifprj_relaxed.py
    * run JGDID algorithm for inlufence diagram
    * input files are stored in problems_id

  * run_gddhingeshifprj_relaxed.py
    * run JGDID algorithm for inlufence diagram
    * input files are stored in problems_relaxed

  * run_gddmixed.py
    * run gdd algorithm for mixed inference (MMAP with interleaved max and sum)
    * input files are stored in problems_mixed
    
  * run_weightedmbe.py
    * run weighted mini bucket elimination for influence diagram
    * input files are stored in problems_id

* generating problems
  * gen_id_from_bn.py
    * geneate random influence diagram from Bayes net
    * Bayes nets and random influence diagrams are stored in beta

  * gen_mdp.py
    * generate random factored mdp
    * generated influence diagrams (uai, id, pvo) and gpickle files are stored in beta
  
  * gen_pomdp.py
    * generate random factored pomdp
    * generated influence diagrams (uai, id, pvo) and gpickle files are stored in beta

  * gen_sysadmin.py
    * generated sysadmin factored mdp problems


* converting formats
  * trans_erg_to_uai.py
    * convert ERG-UAI (erg) to ID-UAI (uai, mi, pvo)
  
  * trans_id_to_mixed.py
    * convert ID-UAI (uai, id, pvo) to MI-MMAP (uai, mi, pvo) format
  
  * trans_limid_to_mmap.py
    * convert LIMID (*.limid) to standard uai MMAP (uai, map) format

  * trans_mmap_to_mixed.py
    * convert uai MMAP (uai, map) standard format to MI-MMAP (uai, mi, pvo) mixed format
   
  * trans_msis.py
    * relax input influence diagram (input, output in ID-UAI format)
    * generates relaxed partial variable ordering *.relaxed.pvo and *.png file showing relaxed influence diagram

  * trans_uai_to_limd.py 
    * converts UAI format (.uai, .pvo, .id) to limid format
 
* draw graphs
  * draw_sysadmin.py

  

# TODO
* upgrade to to python3 
* upgrade code to recent networkx libs
* use logger

