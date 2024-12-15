#ifndef __NODE__
#define __NODE__

#include <vector>
#include <string>
#include <map>

#include "simple_tensor.h"


class Node {
    public:
        Node(SimpleTensor* value_ptr, bool is_input=true) : _is_input{is_input} {
            _input = value_ptr;
            _node_id = (std::string) (*value_ptr);
            node_num++;
        };

        SimpleTensor* getValue() { return _input; };

        const SimpleTensor* getValue() const { return _input; };

        std::string getId() const { return _node_id; };

        const std::vector<Node*> getChildren() const { return _children; };

        const std::vector<Node*> getParents() const { return _parents; };

        SimpleTensor* getWholeGradValue() { return _whole_gradient_value; };

        std::map<std::string, SimpleTensor*> getLocalGradValues() { return _local_gradient_values; };

        std::string getOperation() { return _operation; };

        void setChildren(std::vector<Node*> children) { _children = children; };
        
        void setParents(std::vector<Node*> parents) { _parents = parents; };
        
        void setOperation(std::string operation) { _operation = operation; };
        
        void setWholeGradValue( SimpleTensor* value ) { _whole_gradient_value = value; };

        void addParent(Node* parent) { _parents.push_back(parent); };
        
        void addChild(Node* child) { _children.push_back(child); };
        
        void addLocalGradValue(std::string tensor_name, SimpleTensor* local_grad) { 
            // std::cout << "inserting: " << tensor_name << " in node:" << _node_id <<  "\n"; 
            _local_gradient_values.insert({tensor_name, local_grad});
        };

        bool isInput() { return _is_input; }
        
        bool hasParent() { return _parents.size() > 0; }
        
        bool hasChildren() { return _children.size() > 0; }

    private:
        SimpleTensor* _input;
        std::vector<Node*> _children;
        std::vector<Node*> _parents;
        std::string _operation;
        std::map<std::string, SimpleTensor*> _local_gradient_values;
        SimpleTensor* _whole_gradient_value;

        static int node_num;
        std::string _node_id;
        bool _is_input;
};


std::ostream& operator<< (std::ostream& os, const Node node);
#endif
