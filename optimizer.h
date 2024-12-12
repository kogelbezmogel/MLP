#ifndef __OPTIMIZER__
#define __OPTIMIZER__

#include "simple_tensor.h"
#include "model.h"

#include <string>
#include <map>

class Optimizer {
    public:
        Optimizer(Model& model, float lr=0.001) : _lr(lr) {
            std::vector<std::string> layer_names = model.getLayers();
            for(std::string layer_name : layer_names) {
                _parameteres.insert({model[layer_name]->getWeight(), model[layer_name]->getWeight()});
                if( model[layer_name]->bias() )
                    _parameteres.insert({model[layer_name]->getBias(), model[layer_name]->getBias()});
            }
            _graph = model.getGraph();
        }

        void step() {
            // std::cout << "making step";
            for(std::pair<std::string, SimpleTensor> pair : _parameteres) {
                pair.second += (-_lr) * _graph->getNode(pair.second)->getWholeGradValue();
            }
        }

    protected:
        float _lr;
        std::map<std::string, SimpleTensor> _batch_history;
        std::map<std::string, SimpleTensor> _parameteres;
        Graph* _graph;
};


class SGD : public Optimizer {

};

class BGD : public Optimizer {

};

class AdaGrad : public Optimizer {

};

class AdaDelta : public Optimizer {

};

class Adam : public Optimizer {

};

#endif // __OPTIMIZER__