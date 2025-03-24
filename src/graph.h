#ifndef __GRAPH__
#define __GRAPH__

#include <iostream>
#include <functional>
#include <vector>
#include <map>
#include <fstream>
#include <string>

#include "node.h"
#include "simple_tensor.h"

class Graph {
    private:
        std::vector<Node*> _nodes;
        std::vector<Node*> _nodes_in_order;
        std::map<std::string, Node*> _nodes_map;

        Graph( const Graph& graph) { };
    
    public:
        Graph();

        ~Graph();

        std::vector<Node*> getNodes() { return _nodes; }

        void addNode(Node* node_ptr);

        Node* getNode(std::string tensor_name);

        void backwards();

        // Node* operator[] (std::string tensor_name);

        void orderNodesRec( std::map<Node*, bool>& visited, Node* node);

        // void printGraph();

        void clearSequence();

        bool orderNodes();

        bool contains(std::string tensor_name) const {
            return _nodes_map.count(tensor_name);
        }

        // const std::vector<Node*> get_nodes() const { return _nodes; };

        void saveGraphToFile(std::string file_path);

        static void addNodeToGraph(std::vector<SimpleTensor*> args, std::map<std::string, SimpleTensor> derivatives, SimpleTensor* t3_simple, Graph* graph_ptr, std::string operartion);
};



// std::ostream& operator<< (std::ostream& os, const Graph graph);


#endif //__GRAPH__