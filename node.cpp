#include <iostream>
#include "node.h"

std::ostream& operator<< (std::ostream& os, const Node node) {
    os << node.getId() << "(";
    // print_size(os, node.getValue().getSize());
    os << node.getValue();
    os << ")\n";

    // os << "parents: \n";
    // for(const auto* p_ptr : node.getParents() )
    //     os << "-- n" << (*p_ptr).get_id() << "(" << (*p_ptr).get_value() << ")\n";
    // os << "children: \n";
    // for(const auto* c_ptr : node.getChildren() )
    //     os << "-- n" << (*c_ptr).get_id() << "(" << (*c_ptr).get_value() << ")\n";
    return os;
}