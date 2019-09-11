#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

//using boost library
#include <boost/dynamic_bitset.hpp>

//functions
#include "update.h"
#include "mainf.h"

//including class
#include "class_Node.h"
#include "class_Tree.h"
#include "class_Params.h"





void  Update_gRAVE(vector < pair <double, int> > &gRAVE, const boost::dynamic_bitset<> &F, const double &reward_V)
// update gRAVE score for each feature of feature subset F, by adding the reward_V the the score
{
    int f = F.size()-1;  //number of features (without the stopping feature)
    int d = F.count()-1;   //number of features in the subset, without the stopping feature

    //F is should be final when we update gRAVE
    if (!F[f])
    {
        cout << "Error, the feature subsest should be final when updating gRAVE" << endl;
        exit(1);
    }

    for (int fi=0; fi<f; fi++){
        if(F[fi]){
            gRAVE[fi].first = (gRAVE[fi].first*gRAVE[fi].second + reward_V)/(gRAVE[fi].second +1 );
            gRAVE[fi].second ++;
        }
    }

    // Updating the gRAVE score of stopping feature
    gRAVE[f+d].first = (gRAVE[f+d].first*gRAVE[f+d].second + reward_V)/(gRAVE[f+d].second +1 );
    gRAVE[f+d].second ++;

}
















//========================================================================================================================================================================================//

int  Update_Tree_And_Get_Adress(Tree &T, const boost::dynamic_bitset<> &F, const Params &params)
//Returning the address of the node corresponding to feature F
//Adding the node F to the search Tree:
    //-> if the node F is not in the tree, it is added at the end
    //-> if the node F is already in the tree, it does nothing

{
    //finding the index of the feature using hash function (hash function because complexity = O(1))
        string Fl;
        to_string(F, Fl);
        int last_index = T.N_address.size();
        T.N_address.insert({Fl,last_index}); //this will ad the new node if doesn't exist, or do nothing otherwise (looking at the first argument)
        unordered_map<string,int>::const_iterator got = T.N_address.find(Fl); //getting the iterator corresponding to the feature number
        int node_address = got->second; //getting the address from the iterator

    if (node_address == T.N.size())
    {
        T.N.push_back(Node(F,node_address));
        Add_Familly(T,F,node_address);

        T.N[node_address].fs[1] = reward_full(F, params);
    }

    return node_address;
}






//========================================================================================================================================================================================//

void  Add_Familly(Tree &T, boost::dynamic_bitset<> F, const int &node_address)

   //------------------------------------------------------------------------------------------//
   //------------------------- Checking for parents and children nodes ------------------------//
   //------------------------------------------------------------------------------------------//

// If they exist, the address are updated
// here F is send as a copy and not as a pointer to the address, so we can modify it without affecting the global F
{

    string Fp; //parent node string
    string Fc; //child node string
    int parent_node_address;
    int child_node_address;
    unordered_map<string,int>::const_iterator got;

    //indicate to my parent and my child nodes that I exist
    for (int fi=0; fi<F.size(); fi++)
    {
        if (F[fi]) //if this is a 1, replace it with a 0 to find a parent
        {
            F[fi] = 0;
            to_string(F, Fp);
            got = T.N_address.find(Fp);
            if (got != T.N_address.end()) //this parent is int the tree, lets tell him that we exist
            {
                parent_node_address = got->second;
                T.N[parent_node_address].address_f[fi] = node_address;
                T.N[parent_node_address].allowed_features.insert(fi);
            }
            F[fi] = 1; //put the value back to 1 for next loops
        }
        else // if this is a 0, replace it with a 1 to find a child
        {
            F[fi] = 1;
            to_string(F, Fc);
            got = T.N_address.find(Fc);
            if (got != T.N_address.end()) //if my child exist in the tree
            {
                child_node_address = got->second;
                T.N[node_address].address_f[fi] = child_node_address;
                T.N[node_address].allowed_features.insert(fi);

                Update_node_av_var(T.N[node_address],T.N[child_node_address]);
                T.N[node_address].Score_f[fi] = T.N[child_node_address].T_F;


            }
            F[fi] = 0; //put the value back to 1 for next loops
        }
    }


}




void  Update_node_av_var(Node &node, const Node &chnode)
//adding the information of the child nodes to the parent node
{
    double N = (double)node.T_F;
    double K = (double)chnode.T_F;

    double oldMu = node.av_F;
    double MuK = chnode.av_F;

    double Sg = node.sg_F;
    double SgK = chnode.sg_F;


    double newMu = (N*oldMu + K*MuK) / ( N + K );  //compute the new average
    double d1 = oldMu - newMu;
    double d2 = MuK - newMu;
    double NewSg = sqrt(   ( N*(pow(Sg,2.0)+pow(d1,2.0)) + K*(pow(SgK,2.0)+pow(d2,2.0)) ) / ( N + K )   );  //for variance calculation, see Sangita answer at https://www.researchgate.net/post/How_do_I_combine_mean_and_standard_deviation_of_two_groups


    node.T_F = (int)N + int(K);
    node.av_F = newMu;
    node.sg_F = NewSg;

}







