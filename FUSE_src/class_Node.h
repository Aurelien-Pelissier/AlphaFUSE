#ifndef DEF_NODES
#define DEF_NODES

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

//using boost library
#include <boost/dynamic_bitset.hpp>



class Node
{
    public:

    //Builders & Destroyers
    Node(boost::dynamic_bitset<> F, int node_address);
    ~Node();

    //Attributes
    int address;                                    //Node address in Tree T
    boost::dynamic_bitset<> sub_F;                  //Features subset (binary)
    int F_size;                                     //Number of features in the subset
    double T_F;                                        //Nb of time the node has been visited (with exp backpropagation)
    int Tt_F;                                       //Nb of time node has been visited (through UCB phase))
    double av_F;                                    //Average of the node
    double Score_av;                                //Average with weigth adjustement
    double sg_F;                                    //Variance of the node
    std::unordered_set<int> allowed_features;       //Allowed features (only for Discrete heuristic)
    std::vector<int> address_f;                     //Address of the child nodes
    std::vector<std::pair<double, int>> lRAVE_f;    //Local RAVE score of each feature
    std::vector<double> Score_f;                    //Score of the child nodes (for UCB)
    std::vector<double> fs;                         //all informations about stopping feature (tf, muf, sgf)
    std::vector<double> fr;                         //all informations about random exploration from the node (tf, muf, sgf)
    bool tobe_updated;                              //just a tool for the back-propagation
    bool already_updated_score;                     //just a tool for the back-propagation
    double weight;                                  //just a tool for the back-propagation

    //Methods

};




#endif
