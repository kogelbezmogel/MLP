#ifndef __TENSOR_OPERATIONS__
#define __TENSOR_OPERATIONS__

#include "tensor.h"


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
        static std::map<std::string, SimpleTensor> addDerivatives(SimpleTensor t1, SimpleTensor t2, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> mseLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> cceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);
        static std::map<std::string, SimpleTensor> bceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3);

        static act_fun activation(std::string activation) { return _activation_map[activation]; }
        static loss_fun loss(std::string loss) { return _loss_map[loss]; }

    private:
        static Graph* resolveGraphContext(std::vector<Tensor*> args);
        
        static std::map<std::string, act_fun> _activation_map;
        static std::map<std::string, loss_fun> _loss_map;

};


#endif