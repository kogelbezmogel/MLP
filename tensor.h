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


using act_fun = Tensor(*) (Tensor t1);
using loss_fun = Tensor(*) (Tensor t1, SimpleTensor t2);

class TensorOperations {
    public:
        static Tensor add(Tensor t1, Tensor t2);
        static Tensor mul(Tensor t1, Tensor t2);
        static Tensor relu(Tensor t1);

        static Tensor mseLoss(Tensor predicted, SimpleTensor real);
        static Tensor cceLoss(Tensor predicted, SimpleTensor real);
        static Tensor bceLoss(Tensor predicted, SimpleTensor real);

        static std::map<std::string, SimpleTensor> reluDerivatives(SimpleTensor t1, SimpleTensor t2);
        static std::map<std::string, SimpleTensor> mulDerivatives(SimpleTensor t1, SimpleTensor t2, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> mseLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> cceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> bceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);

        static act_fun activation(std::string activation) { return _activation_map[activation]; }
        static loss_fun loss(std::string loss) { return _loss_map[loss]; }

    private:
        static std::map<std::string, act_fun> _activation_map;
        static std::map<std::string, loss_fun> _loss_map;
};


// std::ostream& operator<< (std::ostream& os, const Tensor& tensor);

#endif //__TENSOR__