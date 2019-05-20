# FUSE-for-Raman-Spectroscopy
This repository contains the source code to perform feature selection with a reinforcement learning approach, where the feature set state space is represented by a Direct Acyclic Graph (DAG). It provides a precompiled `C++` implementation of the FUSE 2 algorithm [1], that is an improved version of the FUSE algorithm [2].

&nbsp;



## Running the code
#### Requirements
The algorithm is precompiled and can be run directly from the python script `FUSE.py` available in the `\FUSE` folder (requires `Python 3.5`). The algorithm was compiled with a `C++` compiler in Code::Blocks, any modification of the core algorithm (in `\FUSE_src`) will require to recompile the code (requires the `boost` library (available at https://www.boost.org/) and a `c++14` compiler).

#### Datasets
The dataset involve two files, the `.data` file contains a matrix `[n * f]`,  where *n* is the number of training example and *f* the number of features. the `.labels` file is an array `[f]` that corresponds to the class label associated to each example.

#### Output
Each 2000 iteration, the program display the current candidate for the best feature subset (considered to be the path with highest average at the end of the search), along with its corresponding score:
`
Path with Highest average:
  [ 431, 214, 543 ] -> 0.9879`
  
 At the end of the search (when the maximal number of iteration is reached), the program write the best feature subset in `BigResults.txt`, and all detailed informations in output files `Output_Tree.txt`, `Output_Reward.txt` and `Result.txt`.
(By running the script `plot_reward.py`, we can plot the evelution of the reward during the search)


&nbsp;



## References

[1]: A.  Pelissier,  "Feature Selection as Reinforcement Learning Applied to Raman Spectra for Cancer Diagnosis", Imperial University of Hokkaido, June 2018 [https://www.docdroid.net/L8GOTrg/feature-selection-aurelien-pelissier.pdf]

[2]: Gaudel, Romaric, and Michele Sebag. "Feature selection as a one-player game." International Conference on Machine Learning. 2010. [https://hal.inria.fr/inria-00484049/document].


