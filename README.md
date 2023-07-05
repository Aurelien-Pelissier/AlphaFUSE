# AlphaFUSE  -  Feature-Selection-as-Reinforcement-Learning

<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/AlphaFUSE/master/img/latt.png" width=400>

Dataset often contains many features that are either redundant or irrelevant, and can thus be removed without incurring much loss of information. Decreasing the number of feature have the advantage of reducing overfitting, simplifying models, and also involve shorter training time, which makes it a key aspect in machine learning. 

This repository contains the source code to perform feature selection with a reinforcement learning approach, where the feature set state space is represented by a Direct Acyclic Graph (DAG). It provides the C++ implementation of and improved Monte Carlo DAG Search algorithms. alphaFUSE [1], starts from the empty feature subset and relies on UCT to identify the best feature subest in the DAG with a fixed budget, The information is efficiently backpropagated through all the ancestor nodes after each iteration for an optimal gain of information.


&nbsp;



## Running the code

#### Requirements
The algorithm can be run directly with `alphaFUSE/Alpha_FUSE.py` for Windows user.

The source code is available in the `alphaFUSE_src/` folder, compiling the code requires the `boost` library (available at https://www.boost.org/) and a `c++14` compiler. The .exe file was generated with Code::Blocks IDE, the project can be open with `alphaFUSE_src/Feature_Selection.cbp`.

#### Datasets
alphaFUSE uses 2 files for the dataset. the first one (.data), `L[n][f]` is a matrix where *n* is the number of training example and *f* the number of features. the second one (.labels), is a column array containing the labels for each example in the dataset, anything related to the reading of the dataset is implemented in `dataset.cpp`.


#### The feature set space and stopping feature
<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/AlphaFUSE/master/img/FS.png" width=225>
We define a graph for which each node correspond to a feature set F, adding a feature to this feature set lead to a child node, and removing a feature lead to a parent node. The root of the graph is the empty feature subset. For each node, we also consider the stopping feature fs, which allows the search to stop at the current node instead of adding new features.



&nbsp;



## alphaFUSE Algorithm details

<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/AlphaFUSE/master/img/MCTS.png" width=200>

FUSE relies on the well known UCT algorithm to perform Monte Carlo seach in the feature DAG, and stop when the given number of iteration is reached. At the end of the search, the recommended feature subset is the one at the end of the path with highest average.

### The four phases of one iteration

#### UCT phase
for a node *F*, the selected child *f* node is the one maximizing its UCB Score:
<img src="https://raw.githubusercontent.com/Aurelien-Pelissier/AlphaFUSE/master/img/UCB.png" width=400>  
*TF* is the number of visit of node *F*, and due to the high branching factor of the tree, the exploration is limited to an *Allowed feature* set, which restrict the number of considered child nodes depending of *TF*. A new child node is added whenever int\[*TF*^*b*\] is incremented. 

To know which feature to add, we consider the one maximizing its RAVE score, which depends on the average reward over all final node *F* containing *f*.


#### Random phase

When a node with *TF*=0 is reached, we evaluate the node by performing random exploration until the stopping feature *fs* is added. The probability of chosing  the stopping feature at depth *d* is set to 1-*q*^*d*.

#### Reward Calculation

Once the stopping feature has been selected, the exploration stops and the reward is computed based on a k-Nearest-Neighboor (kNN) classifier. The advantage of kNN is that it requires no prerequisite training, and is not too computationally expensive. The complexity of the reward calculation is scaling as O(*n*^2\**f*/*r*) and is limiting the algorithm to dataset with less than 10000 examples.

#### Backpropagation phase

The original FUSE [2] algorithm backpropagate the reward only for the nodes withing the current path. In alphaFUSE [1], the reward is backpropagated to all ancestor nodes. Since the number of nodes to be updated at depth *d* scales as *2^d*, the backpropagation is said to be exponential. te parameter *alpha* tune the backpropagation of the ancestor nodes, see the original paper [1] for details.

<img align="center" src="https://raw.githubusercontent.com/Aurelien-Pelissier/AlphaFUSE/master/img/backpropagation.png" width=700>


### Simulation parameters
The main simulation parameters can be changed in `src/Main.cpp`.

```c++
    alpha = 1;     //Transposition parameter
    Nt = 100000;   //Number of iterations of the simulation
    q = 0.98;      //Random expansion parameter, used to control the average depth in the random phase, |q|<1
    k = 5;         //Number of nearest neighbors involved in the reward calculation
    m = 50;        //Size of the small subsample for the reward calculation
    ce = 0.5;      //UCB exploration control parameter
    cl = 200;      //l-RAVE/g-RAVE weight in RAVE score
    b = 0.2;       //Discrete heuristic exploration parameter, |b|<1
    
    L[n][f+1] = read_dataset("dataset.dat");  //Training set matrix
```
For details about the parameters, please refer to the implementation details described in [1,2].

### Output

When the simulation is finished, the program mainly return :

- The best feature subset (considered to be the path with highest average at the end of the search)

`
Path with Highest average:
  [ 4-0.9605  2-0.9802  5-0.9823  fs-0.9879 ]`
- The reward and depth search after each iteration
By running `plot_reward.py` (requires `Python 3`), we can plot the evelution of the reward during the search:

All the informations are available in output files `Output_Tree.txt`, `Output_Reward.txt` and `Result.txt`.

- The python interface allows alphaFUSE to be run more several times and to average the result over different seeds. The obtained features after each FUSE execution is saved in `BigResults.txt`




## References

[1]: A.  Pelissier,  A.  Nakamura.   "Backpropagation Phase in Upper Confidence Bound for Single Rooted Directed Analytic Graph". *Unpublished*. 2019. [Draft](https://github.com/Aurelien-Pelissier/AlphaFUSE/blob/master/Alpha_UCD.pdf).

[2]: Gaudel, Romaric, and Michele Sebag. "Feature selection as a one-player game." *International Conference on Machine Learning*. 2010. [https://hal.inria.fr/inria-00484049/document].
