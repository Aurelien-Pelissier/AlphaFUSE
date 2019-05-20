#ifndef INCLUDED_UPDATEH
#define INCLUDED_UPDATEH

#include <vector>
#include <unordered_map>

//using boost library
#include <boost/dynamic_bitset.hpp>

//including classes
#include "class_Node.h"
#include "class_Tree.h"
#include "class_Params.h"


//Update
void                                  Update_gRAVE(std::vector <std::pair<double,int>> &gRAVE, const boost::dynamic_bitset<> &F, const double &reward_V);
int                                   Update_Tree_And_Get_Adress(Tree &T, const boost::dynamic_bitset<> &F, const Params &params);
void                                     Add_Familly(Tree &T, const boost::dynamic_bitset<> F, const int &node_address);
void                                     Update_node_av_var(Node &node, const Node &chnode);

//backpropagation
void                                  Backpropagate(Tree &T, Node &node, const int &fi, const boost::dynamic_bitset<> &Ft, const double &reward_V );
void                                  Check_Ancestors(Tree &T, Node &node);
void                                  Compute_Weights(Tree &T, Node &node, double current_weight);
void                                  Update_Ancestors(Tree &T, Node &node, const int &f_last, const boost::dynamic_bitset<> &Ft, const double &reward_V );

void                                  Update_Node(Tree &T, Node &node, const int &fi, const boost::dynamic_bitset<> &Ft, const double &reward_V);
void                                  Update_av_var(double &N_IT, double &AV, double &SG, const double &xN1, const double &weight);
void                                  Update_av_var(int &N_IT, double &AV, double &SG, const double &xN1, const double &weight);

#endif
