#include <iostream>
#include <functional>
#include <vector>
#include <map>
#include <fstream>

#include "graph_tensor.h"
#include "tensor.h"
#include "utils.h"

int Node::node_num = 0;

Graph::Graph() {
};

void Graph::backwards() {

    for(Node* node : _nodes) {
        std::cout << node -> getId() << " | children: ";
        for(Node* child: node -> getChildren())    
            std::cout << child -> getId() << " ";
        std::cout << "| parents: ";
        for(Node* parent: node -> getParents())    
            std::cout << parent -> getId() << " ";
        std::cout << "|\n";
    }

    for(Node* node : _nodes) {
        std::cout << *(node -> getValue()) << "\n\n";
    }

    

    //sotrgin nodes
    orderNodes();

    // setting local_grad and grad of last node to 1
    Node* last_node = _nodes_in_order[ _nodes_in_order.size()-1 ];

    std::cout << "last " << last_node->getId() << " size: " <<  str_representation( last_node->getValue()->getSize() ) << "\n";
    last_node -> setWholeGradValue( new SimpleTensor(last_node->getValue() -> getSize(), 1.0) );

    SimpleTensor* grad_value;
    for(std::vector<Node*>::reverse_iterator ite = _nodes_in_order.rbegin() + 1; ite != _nodes_in_order.rend(); ite++) {
        std::cout << (*ite)->getId() << ":   0";
        grad_value = new SimpleTensor();
        for(Node* child : (*ite) -> getChildren() ) {
            (*grad_value) += (*((*ite)->getLocalGradValues()[child->getId()])) * (*(child->getWholeGradValue()));
            std::cout << " + mul(" 
                      << str_representation((*ite)->getLocalGradValues()[child->getId()] -> getSize())
                      << " * "
                      << str_representation(child->getWholeGradValue() -> getSize())
                      << ")";
        }
        grad_value -> trim();
        std::cout << " = " << str_representation( grad_value -> getSize() ) << "\n";
        (*ite) -> setWholeGradValue(grad_value);
    }
}

bool Graph::addNode(Node* node_ptr) {
    _nodes.push_back(node_ptr);
    std::cout << "node " << node_ptr->getId() << ": " << node_ptr << "\n";
    _nodes_map.insert( { node_ptr->getId(), node_ptr} );
    // std::cout << "adding node: " << node_ptr->getId() << " " << str_representation(node_ptr->getValue().getSize()) << "\n";
    return true;
}


Node* Graph::operator[] (std::string tensor_name) {
    return _nodes_map[tensor_name];
}

Node* Graph::getNode (std::string tensor_name) {
    return _nodes_map[tensor_name];
}


void Graph::orderNodesRec( std::map<Node*, bool>& visited, Node* node) {
    // to mark node as visited its parents need to be marked before
    bool parents_visited = true; 
    for(Node* parent_node : node->getParents())
        parents_visited *= visited[parent_node];
    if(parents_visited && visited[node] == false) {
        visited[node] = true;
        _nodes_in_order.push_back(node);
        
        // recursive function call
        for(Node* child_node : node->getChildren())
            orderNodesRec(visited, child_node);
    }
}

void Graph::clearSequence() {
    orderNodes();

    // std::cout << "\nbefore clearing\nnodes: ";
    // for(Node* node : _nodes)
    //     std::cout << node -> getId() << " | ";
    // std::cout << "\nnodes_in_order: ";
    // for(Node* node : _nodes_in_order)
    //     std::cout << node -> getId() << " | ";
    // std::cout << "\n clearing: ";

    _nodes.clear();
    
    for(std::vector<Node*>::reverse_iterator node_ite = _nodes_in_order.rbegin(); node_ite < _nodes_in_order.rend(); node_ite++)
        if( (*node_ite) -> isInput() ) {
            // std::cout << "-" << (*node_ite) -> getId() << "\n";
            delete (*node_ite);
        } else
            _nodes.push_back( (*node_ite) );
    _nodes_in_order.clear();
    
    for(Node* node : _nodes)
        node -> setChildren({});

    // std::cout << "\n\nafter clearing\nnodes: ";
    // for(Node* node : _nodes)
    //     std::cout << node -> getId() << " | ";
    // std::cout << "\nnodes_in_order: ";
    // for(Node* node : _nodes_in_order)
    //     std::cout << node -> getId() << " | ";
    // std::cout << "\n\n";
}

bool Graph::orderNodes() {
    std::map<Node*, bool> vistited;
    _nodes_in_order.clear();

    for(auto* node : _nodes)
        vistited.insert({node, false});

    for(auto* node: _nodes)
            orderNodesRec(vistited, node);

    // printing resulkts
    // std::cout << "start -> \n";
    // for(Node* node : _nodes_in_order)
    //     std::cout << "n" << node->getId() << " -> ";
    // std::cout << "\nend\n";
    return true;
}

void Graph::saveGraphToFile(std::string file_path) {

    // std::map<std::string, std::string> names_map;
    // for(int i = 0; i < _nodes.size(); i++)
    //     names_map.insert({_nodes[i]->getId(), "n" + std::to_string(i)});

    std::ofstream fout(file_path);
    fout << "digraph { \n";
    for(Node* node: _nodes) {
        if(node -> getOperation() != "")
            fout << "\"" << node->getId()<< "\"[xlabel=\"" << node -> getOperation() << "\"]\n";
        for(Node* node_child: node->getChildren())
            fout << "\"" <<  node->getId() << "\" -> \"" << node_child->getId() << "\"\n";
    }
    fout << "}";
}

std::ostream& operator<< (std::ostream& os, const Graph graph) {
    for(const auto* n_ptr : graph.get_nodes() )
        os << str_representation( n_ptr->getValue() -> getSize() );
    return os;
}