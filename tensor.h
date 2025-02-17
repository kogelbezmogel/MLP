#ifndef __TENSOR__
#define __TENSOR__

#include <iostream>
#include <vector>
#include <cstdarg>
#include <algorithm>
#include <string>

#include "graph.h"
#include "simple_tensor.h"


class Tensor : public SimpleTensor {

    friend class TensorOperations;

    public:
        Tensor(): SimpleTensor() { }

        // Tensor(std::vector<size_t> size, float init, bool calcGrad = false, Graph* graphContext = nullptr);

        Tensor(const SimpleTensor& SimpleTensor, bool calcGrad = false, Graph* graphContext = nullptr);

        Tensor(std::vector<size_t> size, std::vector<float> data, bool calcGrad = false, Graph* graphContext = nullptr, bool _is_input = true);


        // Tensor(std::vector<size_t> size, float* data, bool calcGrad = false, Graph* graphContext = nullptr);

        // Tensor(bool calcGrad = false, Graph* graphContext = nullptr);

        // Tensor(float value, bool calcGrad = false, Graph* graphContext = nullptr);

        // Tensor(const Tensor& to_copy) : SimpleTensor(to_copy) {
        //     _calc_grad = to_copy._calc_grad;
        //     _grad_graph = to_copy._grad_graph;
        // }

        // Tensor(Tensor&& to_move);

        void setGrapContext(Graph* graph_ptr);

        Tensor& operator=(const Tensor&);
        
        // Tensor& operator=(Tensor&& to_move);

        Graph* getGraphContext() { return _grad_graph; }

        void setCalcGrad(bool val = true);

    private:
        
        bool _calc_grad {false};
        Graph* _grad_graph {nullptr};
};

// std::ostream& operator<< (std::ostream& os, const Tensor& tensor);

#endif //__TENSOR__