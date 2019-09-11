#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>

//using boost library
#include <boost/dynamic_bitset.hpp>
#include "kdtree/alglibmisc.h"
#include "kdtree/stdafx.h"

//functions
#include "update.h"
#include "select_features.h"
#include "mainf.h"

//including class
#include "class_Node.h"
#include "class_Tree.h"

using namespace std;
using namespace alglib;







void   Backpropagate(Tree &T, Node &node, const int &fi, const boost::dynamic_bitset<> &Ft, const double &reward_V, const Params &params)
//backpropagate the statistic from the leaf node to all of the ancestors
{
    double alpha = params.alpha;
    Check_Ancestors(T, node);
    Compute_Weights(T, T.N[0], 1.0, alpha);
    Update_Ancestors(T, T.N[0], fi, Ft, reward_V );
}




void   Check_Ancestors(Tree &T, Node &node)
// recursive function which check all the ancestors to be updated
{

    //tag myself
    node.tobe_updated = true;


    //Now tag my parents
    if (node.sub_F.count() == 0) return; //if we arrived to the ultimate ancestor (empty Feature set), we can move on

    boost::dynamic_bitset<> F = node.sub_F;
    string Fp;
    int parent_node_address;
    unordered_map<string,int>::const_iterator got;

    for (int i=0; i<F.size(); i++)
    {
        if (F[i]) //if this is a 1, replace it with a 0 to find a parent
        {
            F[i] = 0;
            to_string(F, Fp);
            got = T.N_address.find(Fp);
            if (got != T.N_address.end()) //we found it, lets update him
            {
                parent_node_address = got->second;
                if (T.N[parent_node_address].tobe_updated == false)
                {
                    Check_Ancestors(T, T.N[parent_node_address]);
                }
            }
            F[i] = 1; //put the value back to 1 for next loops
        }
    }

    return;

}


void Compute_Weights(Tree &T, Node &node, double current_weight, double alpha)
//Compute the weight of each nodes who need to be updated
{
    int f = node.sub_F.size()-1;

    // check the number of child in the bakcpropagation path
    double n_chupd = 0;
    for ( auto &fi : node.allowed_features )
    {
        if ((fi!=f) && (T.N[node.address_f[fi]].tobe_updated == true))
        {
            n_chupd++;
        }
    }

    if (n_chupd == 0) //it means that stopping feature or random exploration has been performed-> It is a leaf node
    {
        node.weight = 1.0;
        return;
    }


    double ch_weight = current_weight/n_chupd;
    for ( auto &fi : node.allowed_features )
    {
        if ((fi!=f) && (T.N[node.address_f[fi]].tobe_updated == true))
        {
            T.N[node.address_f[fi]].weight += ch_weight;
            Compute_Weights(T, T.N[node.address_f[fi]], ch_weight, alpha);
        }
    }
}




/*
void Compute_Weights(Tree &T, Node &node, double current_weight, double alpha)
//Compute the weight of each nodes who need to be updated
{
    int f = node.sub_F.size()-1;

    int n_child = 0;
    double ch_weight = 0;
    for ( auto &fi : node.allowed_features )
    {
        if ((fi!=f) && (T.N[node.address_f[fi]].tobe_updated == true))
        {

            if (T.N[node.address_f[fi]].selected_through_descent == true)
            {
                T.N[node.address_f[fi]].weight = 1.0;
                //cout << "descent" << endl;
            }
            else
            {
                T.N[node.address_f[fi]].weight = alpha;
                //cout << "not descent" << alpha << endl;
            }
            Compute_Weights(T, T.N[node.address_f[fi]], ch_weight, alpha);
            n_child ++;
        }
    }

    if (n_child==0) //it means that stopping feature or random exploration has been performed -> It is a leaf node
    {
        node.weight = 1.0;
        return;
    }
}
*/


void Update_Ancestors(Tree &T, Node &node, const int &f_last, const boost::dynamic_bitset<> &Ft, const double &reward_V )
{

    int f = node.sub_F.size()-1;
    bool check_end = true;

    for ( auto &fi : node.allowed_features )
    {
        if ((fi!=f) && (T.N[node.address_f[fi]].tobe_updated == true))
        {
            check_end = false;
            if (T.N[node.address_f[fi]].already_updated_score == false)
            {
                Update_Ancestors(T, T.N[node.address_f[fi]], f_last, Ft, reward_V );
            }
        }
    }

    int f_child = 0;
    if (check_end)
    {
        f_child = f_last;
    }
    Update_Node(T, node, f_child, Ft, reward_V); //We adjust the parent node AFTER that all its children has been updated, (edit -- Not important ??)
    node.already_updated_score == true;
}






void   Update_Node(Tree &T, Node &node, const int &fi, const boost::dynamic_bitset<> &Ft, const double &reward_V)
// Updating the node with inputs:
//  - the node to be updated
//  - the computed reward
//  - the feature that have been selected in the node fi (if -1 it mean that random exploration has been preformed)
//  - the final feature subset Ft (used only for lRAVE score calculation), it's actually not necessary final in the case of random iteration

// the allowed features and the child-node address are not updated here, since they have already been updated in iterate
{
    node.already_updated_score = true;  //mark this node as updated
    double xN1 = reward_V;
    int f = Ft.size()-1;


    //Updating lRAVE scores
    double lRAVE;
    int tl;
    for (size_t i=0; i<f; ++i)
    {
        if (Ft[i]^node.sub_F[i]) //^ is the XOR operation, in our case it correspond to subtracting the vectors : Ft - sub_F, note that sub_F is always included in Ft
        {
            lRAVE = node.lRAVE_f[i].first;
            tl = node.lRAVE_f[i].second;

            node.lRAVE_f[i].first = (lRAVE*(double)tl + xN1)/((double)tl+1);
            node.lRAVE_f[i].second = tl + 1;
        }
    }


    //update the node with appropriate weight
    Update_av_var(node.T_F, node.av_F, node.sg_F, xN1, node.weight);

    if (fi==f)  //updating stopping feature values
    {
        Update_av_var(node.fs[0], node.fs[1], node.fs[2], xN1, 1.0);
        return;
    }


    if (fi==-1)   //updating random exploration information
    {
        Update_av_var(node.fr[0], node.fr[1], node.fr[2], xN1, 1.0);
        return;
    }

}


void  Update_av_var(double &N_IT, double &AV, double &SG, const double &xN1, const double &weight)
// update the average AV and the variance SG with:
//     - the number of iteration already involved N
//     - the value to be added xN1
{
        double N = N_IT;
        double muN = AV;                        //previous average
        double muN1 = (muN*N + xN1*weight)/(N+weight);    //new average
        double sgN = SG;
        double SSN = (xN1 - muN1)*(xN1 - muN);  //for variance calculation, see details at (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
        double sgN1 = sqrt( (SSN*weight + N*pow(sgN,2))/(N+weight) );

        N_IT = N+weight;
        AV = muN1;
        SG = sgN1;
}


