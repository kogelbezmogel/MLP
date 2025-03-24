#include "tensor.h"
#include "utils.h"
#include "except.h"

#include <algorithm>
#include <cmath>


Tensor::Tensor(const SimpleTensor& simple_tensor, bool calc_grad, Graph* graph_context) :
    SimpleTensor(simple_tensor),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
    if(_calc_grad  && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }


// Tensor::Tensor(std::vector<size_t> size, float init, bool calc_grad, Graph* graph_context) :
//     SimpleTensor(size, init),
//     _calc_grad(calc_grad),
//     _grad_graph(graph_context) {
        
//      if(_calc_grad  && !_grad_graph->contains(*this) )
//         _grad_graph -> addNode( new Node(*this) );
//     }

Tensor::Tensor(std::vector<size_t> size, std::vector<float> data, bool calc_grad, Graph* graph_context, bool is_input) :
    SimpleTensor(size, data),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
     
    if(_calc_grad  && !_grad_graph->contains(*this))
        _grad_graph -> addNode( new Node(*this, is_input) );
    }


// Tensor::Tensor(const Tensor& to_copy) : SimpleTensor(to_copy) {
//     std::cout << "copy\n";

//     _calc_grad = to_copy._calc_grad;
//     _grad_graph = to_copy._grad_graph;
// }

void Tensor::setGrapContext(Graph* graph_ptr) {
 _grad_graph = graph_ptr;
}

void Tensor::setCalcGrad(bool val) {
    _calc_grad = val;
}

// Tensor& Tensor::operator=(Tensor&& to_move) {
//     SimpleTensor::operator=(std::move(to_move));
//     std::cout << "moved\n";
//     _calc_grad = to_move._calc_grad;
//     _grad_graph = to_move._grad_graph;

//     to_move._calc_grad = false;
//     to_move._grad_graph = nullptr;
//     return *this;
// }

Tensor& Tensor::operator=(const Tensor& to_copy) {
    SimpleTensor::operator=(to_copy);
    // std::cout << "to_copy\n";
    _calc_grad = to_copy._calc_grad;
    _grad_graph = to_copy._grad_graph;

    return (*this);
}
        

// Tensor::Tensor(std::vector<size_t> size, float* data, bool calc_grad, Graph* graph_context) :
//     SimpleTensor(size, data),
//     _calc_grad(calc_grad),
//     _grad_graph(graph_context) {
        
//     if(_calc_grad  && !_grad_graph->contains(*this) )
//         _grad_graph -> addNode( new Node(*this) );
//     }

        
// std::ostream& operator<< (std::ostream& os, const Tensor& tensor) {
//     std::vector<size_t> size = tensor.getSize();
//     print_rec(os, tensor);
//     os << "(tensor";
//     print_size(os, tensor.getSize());
//     os << ")";
//     return os;
// }
