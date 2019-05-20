#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

//libraries
#include <boost/dynamic_bitset.hpp>

//including functions
#include "mainf.h"
#include "update.h"
#include "print.h"

//including class
#include "class_Node.h"
#include "class_Params.h"
#include "class_Tree.h"




/// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// iterate is a RECURSIVE function, which mean that it is calling itself many times, until it return something.                                            //
/// In our case, the function will call itself until the features subset is final or until it arrived to a new node and perform random exploration.         //
/// The function will then back-propagate and update the nodes until it finally arrive to main again.                                                       //
/// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// - The variables fi and parent_node_address are static so they are the TRANSFERED to each recursive call                                                 //
///   so the informations about the path taken by the exploration is kept in the memory and back-propagation can occur                                      //
/// - The feature subset F is transmitted by ADDRESS so it is updated after each recursive call, at the end F is always final and is used to update l-RAVE  //
/// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double iterate(Tree &T, vector <pair<double,int>> &gRAVE, boost::dynamic_bitset<> &F, const Params &params, int &depth)
// exploring the tree with UCB until:
    // -> It arrive to a new node and perform a random exploration
    // -> It select the stopping feature

// The reward is then calculated and all the nodes score are updated with back-propagation.

{

    double reward_V;
    static int fi = 0; //selected feature
    static int parent_node_address = 0;
    int parent_node_address_rec = 0;

    bool limit_depth = true; //tool to limit the depth of the FUSE search to a specific number
    int depth_limit = 20;


    if (F[F.size()-1])// if Feature subset final
    {
        depth = F.count()-1;
        reward_V = T.N[parent_node_address].fs[1];
        Update_gRAVE(gRAVE, F, reward_V);
        Backpropagate(T, T.N[parent_node_address], fi, F, reward_V);
        //cout << "  final selected" << endl;
    }

    else
    {
        parent_node_address = Update_Tree_And_Get_Adress(T, F, params);
        parent_node_address_rec = parent_node_address;

        if (limit_depth)
        {
            if (F.count()== depth_limit)
            {
                F[F.size()-1] = 1;
                reward_V = iterate (T,gRAVE,F,params,depth);
                return reward_V;
            }
        }

        if (T.N[parent_node_address].T_F != 0) //if node already visited
        // ============================= UCB exploration ============================= //
        {
            // --------------------- Discrete heuristic --------------------- //

            fi = Discrete_UCB_fs(T.N[parent_node_address], gRAVE, T, params);

            // -------------------------------------------------------------- //

            if (fi==-1) //it means that no feature has been selected and that we are going to perform random exploration
            {
                depth = F.count();
                reward_V = iterate_random(T,F,params);
                Update_gRAVE(gRAVE, F, reward_V);
                Backpropagate(T, T.N[parent_node_address], fi, F, reward_V);
                //cout << "  random selected from UCB" << endl;
            }
            else //add the feature to the feature set
            {
                F[fi] = 1;
                reward_V = iterate (T,gRAVE,F,params,depth);
            }
        }


        else //if node never visited before
        // =========================== Random exploration =========================== //
        {
            fi = -1; //indicate that random exploration has been perform and thus no feature selected

            depth = F.count();
            reward_V = iterate_random(T,F,params);
            Update_gRAVE(gRAVE, F, reward_V);
            Backpropagate(T, T.N[parent_node_address], fi, F, reward_V);
            //cout << "  random selected for address " << parent_node_address << endl;
        }
        T.N[parent_node_address_rec].Tt_F++;
        // ========================================================================== //
    }
    return reward_V;

}
