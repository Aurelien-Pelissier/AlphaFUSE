#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>   //pair
#include <map>
using namespace std;

//libraries
#include <boost/dynamic_bitset.hpp>

//include classes
#include "class_Node.h"
#include "class_Params.h"
#include "class_Tree.h"

/* Class Node:
    vector<bool> sub_F;                   //Features subset
    int nF;                               //Number associated to feature subset
    int F_size;                           //Number of features in the subset
    int T_F;                              //Nb of time the node has been visited
    double av_F                           //Average of the node
    double sg_f                           //Variance of the node
    unordered_set<int> allowed_features;  //Allowed features (only for Discrete heuristic)
    vector<int> address_f;                //Node address of the child nodes, (= -1 if never visited)
    vector<pair<double, int>> lRAVE_f;    //Local RAVE score of each feature
    double Node_Score;                    //Score of the node (it is the average at the moment)
    int N_final;                          //Number of time this node has been chosen final
*/




//================================================================================================================================================================//
//====================================================================== Discrete Heuristic ======================================================================//
//================================================================================================================================================================//

int Discrete_UCB(Node &node, const vector <pair<double,int>> &gRAVE, const Tree &T, const Params &params)
// Return the feature corresponding to the highest UCB score
{
    int d = node.F_size;  //feature subset size
    int f = node.sub_F.size()-1; //number of features
    int nrand = 100;
    double T_Fr = (double)node.T_F - (double)nrand;
    if (T_Fr <= 0) T_Fr = 1.0;






    //----------------------------------------------------------------------------------------------------------------------------------------------------------//
    //------------------------------------------- Exploring a new child node whenever floor(T_F^b) is incremented ----------------------------------------------//
    //----------------------------------------------------------------------------------------------------------------------------------------------------------//


    if (node.T_F < nrand) // we perform random exploration fort the first 50 visits
    {
        return -1; //-1 indicate that we don't to want chose any feature
    }







    //if ( ( ( (floor(pow((double)node.T_F+(double)1,params.b)) - floor(pow((double)node.T_F,params.b))) != 0) && (node.allowed_features.size() < (f-d+1)) )  )
    if ( ( ( (floor(pow(T_Fr+1.0,params.b)) - floor(pow(T_Fr,params.b))) != 0) && (node.allowed_features.size() < (f-d+1)) )  )
    {    // if  (                   floor(T_F^b) incremented                       &&                  allowed feature subset not already full  )


        if (node.allowed_features.find(f) == node.allowed_features.end()) //allowed feature set is empty
        {
            node.allowed_features.insert(f);
            return f;
        }


        // Adding the child node with the highest RAVE score to allowed features
        double beta;

        map <double, int>  RAVE;  //map is a SORTED array of element, so the maximum value is found in O(1), storing the RAVE score with its corresponding feature
        double RAVE_Score;
        for (int fi=0; fi<f; fi++)
        {
            if ( (!node.sub_F[fi]) && (node.allowed_features.find(fi) == node.allowed_features.end())  )
            //    if feature is not already selected         &&       fi is not already in allowed features
            {
                beta = params.cl/(params.cl + (double)node.lRAVE_f[fi].second);
                RAVE_Score = (1-beta)*node.lRAVE_f[fi].first + beta*gRAVE[fi].first;
                RAVE.insert({RAVE_Score, fi});
            }
        }
        node.allowed_features.insert(RAVE.rbegin()->second);  //adding (to allowed feature set) the feature corresponding to the maximum calculated RAVE score
                                                                 // (it is the one at the end of the map)
    }


    else //if [TF^b] not incremented, and if stopping feature not added, we force the prgogram to add it
    {
        if (node.allowed_features.find(f) == node.allowed_features.end()) //allowed feature set is empty
        {
            node.allowed_features.insert(f);
            return f;
        }
    }





    //------------------------------------------------------------------------------------------------------------------------------------------------//
    //------------------------------------------------- Computing the UCB scores of allowed features -------------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------------------//

    map <double, int>  UCB; //map is a SORTED array of element, so the maximum value is found in O(1), storing the score with its corresponding feature
    double UCB_Score;

    int T_F;
    int t_f;
    double mu_f;
    double sg_f;

    for ( auto &fi : node.allowed_features ) //computing UCB for each allowed feature
    {
        T_F = node.T_F;

        if (fi==f) //stopping feature
        {
            t_f = node.fs[0];
            mu_f = node.fs[1];
            sg_f = node.fs[2];
        }
        else
        {

            if (node.address_f[fi] == -1) //if feature never selected before, UCB score is infinite (the child node will then be added to the tree)
            {
                return fi;  //return the new feature
            }

            t_f = T.N[node.address_f[fi]].T_F;
            mu_f = T.N[node.address_f[fi]].av_F;
            sg_f = T.N[node.address_f[fi]].sg_F;
        }

        UCB_Score = mu_f + sqrt(    params.ce*log((double)T_F)/(double)t_f  *  min( (double)0.25 ,  (double)pow(sg_f,2) + sqrt(2*log((double)T_F)/(double)t_f) )    );
        UCB.insert({UCB_Score, fi});
    }

    return UCB.rbegin()->second; //return the feature corresponding to the maximum calculated score, (it is the end of the map)

}









//================================================================================================================================================================//
//================================================================== Discrete Heuristic with fs ==================================================================//
//================================================================================================================================================================//

int Discrete_UCB_fs(Node &node, const vector <pair<double,int>> &gRAVE, const Tree &T, const Params &params)
// Return the feature corresponding to the highest UCB score
{
    int d = node.F_size;  //feature subset size
    int f = node.sub_F.size()-1; //number of features
    int nrand = 100;
    double T_Fr = (double)node.T_F - (double)nrand;
    if (T_Fr <= 0) T_Fr = 1.0;






    //----------------------------------------------------------------------------------------------------------------------------------------------------------//
    //------------------------------------------- Exploring a new child node whenever floor(T_F^b) is incremented ----------------------------------------------//
    //----------------------------------------------------------------------------------------------------------------------------------------------------------//


    if (node.T_F < nrand) // we perform random exploration fort the first 50 visits
    {
        return -1; //-1 indicate that we don't to want chose any feature
    }







    //if ( ( ( (floor(pow((double)node.T_F+(double)1,params.b)) - floor(pow((double)node.T_F,params.b))) != 0) && (node.allowed_features.size() < (f-d+1)) )  )
    if ( ( ( (floor(pow(T_Fr+1.0,params.b)) - floor(pow(T_Fr,params.b))) != 0) && (node.allowed_features.size() < (f-d+1)) )  )
    {    // if  (                   floor(T_F^b) incremented                       &&                  allowed feature subset not already full  )


        if (node.allowed_features.find(f) == node.allowed_features.end()) //allowed feature set is empty
        {
            node.allowed_features.insert(f);
            return f;
        }


        // Adding the child node with the highest RAVE score to allowed features
        double beta;

        map <double, int>  RAVE;  //map is a SORTED array of element, so the maximum value is found in O(1), storing the RAVE score with its corresponding feature
        double RAVE_Score;
        for (int fi=0; fi<f; fi++)
        {
            if ( (!node.sub_F[fi]) && (node.allowed_features.find(fi) == node.allowed_features.end())  )
            //    if feature is not already selected         &&       fi is not already in allowed features
            {
                beta = params.cl/(params.cl + (double)node.lRAVE_f[fi].second);
                RAVE_Score = (1-beta)*node.lRAVE_f[fi].first + beta*gRAVE[fi].first;
                RAVE.insert({RAVE_Score, fi});
            }
        }
        node.allowed_features.insert(RAVE.rbegin()->second);  //adding (to allowed feature set) the feature corresponding to the maximum calculated RAVE score
                                                                 // (it is the one at the end of the map)
    }


    else //if [TF^b] not incremented, and if stopping feature is not in allowed feature, we force the search to add the stopping feature
    {
        if (node.allowed_features.find(f) == node.allowed_features.end()) //allowed feature set is empty
        {
            node.allowed_features.insert(f);
            return f;
        }
    }





    //------------------------------------------------------------------------------------------------------------------------------------------------//
    //------------------------------------------------- Computing the UCB scores of allowed features -------------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------------------//

    map <double, int>  UCB; //map is a SORTED array of element, so the maximum value is found in O(1), storing the score with its corresponding feature
    double UCB_Score;

    double T_F;
    double t_f;
    double mu_f;
    double mu_fs;
    double sg_f;

    for ( auto &fi : node.allowed_features ) //computing UCB for each allowed feature
    {
        T_F = node.T_F;

        if (fi==f) //stopping feature
        {
            t_f = node.fs[0];
            mu_f = node.fs[1];
            sg_f = node.fs[2];
			mu_fs = node.fs[1];
        }
        else
        {

            if (node.address_f[fi] == -1) //if feature never selected before, UCB score is infinite (the child node will then be added to the tree)
            {
                return fi;  //return the new feature
            }

            t_f = T.N[node.address_f[fi]].T_F;
            //t_f = node.Score_f[fi];
            mu_f = T.N[node.address_f[fi]].av_F;
            sg_f = T.N[node.address_f[fi]].sg_F;
			mu_fs = T.N[node.address_f[fi]].fs[1];
        }


        //cout << setprecision(4) << max(mu_f,mu_fs) << endl;
        //UCB_Score = max(mu_f,mu_fs) + sqrt(    params.ce*log(T_F)/t_f  *  min( (double)0.25 ,  (double)pow(sg_f,2) + sqrt(2*log(T_F)/t_f) )    );
        UCB_Score = mu_f + sqrt(    params.ce*log(T_F)/t_f  *  min( (double)0.25 ,  (double)pow(sg_f,2) + sqrt(2*log(T_F)/t_f) )    );
        //UCB_Score = max(mu_f,mu_fs) + sqrt(    params.ce*log(T_F)/t_f   );
        UCB.insert({UCB_Score, fi});
    }

    return UCB.rbegin()->second; //return the feature corresponding to the maximum calculated score, (it is the end of the map)

}








