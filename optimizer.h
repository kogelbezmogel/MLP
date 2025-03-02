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

        virtual void step() {}

    protected:
        float _lr;
        std::map<std::string, SimpleTensor> _parameteres;
        Graph* _graph;
};


class SGD : public virtual Optimizer {
    public:
        SGD(Model& model, float lr=0.001) : Optimizer(model, lr) {
        }

        void step() override {
            // std::cout << "making step";
            for(std::pair<std::string, SimpleTensor> pair : _parameteres) {
                pair.second += (-_lr) * _graph->getNode(pair.second)->getWholeGradValue();
            }
        }
};


class BGD : public virtual Optimizer {
    public:
        BGD(Model& model, size_t batch_size, float lr=0.001) : Optimizer(model, lr) {
            _batch_size = batch_size;
            _data_points_counter = 0;

            // initializing _batch history with zeroed tensors
            reset_history();
        }

        void step() override {
            // making step
            if(_data_points_counter >= _batch_size) {
                float coefficient = -_lr  / _data_points_counter;
                for(std::pair<std::string, SimpleTensor> pair : _parameteres) {
                    pair.second += coefficient * _batch_history[pair.second];
                }
                reset_history();
                _data_points_counter = 0;
            // gathering gradients to history
            } else {
                for(std::pair<std::string, SimpleTensor> pair : _parameteres)
                    _batch_history[pair.second] += _graph->getNode(pair.second)->getWholeGradValue();
                _data_points_counter++;
            }
        }

        void close() {
            // makes last step if batch was smaller than desired batch_size
            if(_data_points_counter > 0) {
                float coefficient = -_lr  / _data_points_counter;
                for(std::pair<std::string, SimpleTensor> pair : _parameteres)
                    pair.second += coefficient * _batch_history[pair.second];
                _data_points_counter = 0;
                reset_history();
            }
        }


        void reset_history() {
            // initializing _batch history with zeroed tensors
            _batch_history.clear();
            for(std::pair<std::string, SimpleTensor> pair : _parameteres) {
                _batch_history.insert({pair.first, SimpleTensor(pair.second.getSize(), 0.0)});
            }
        }

    protected:
        size_t _batch_size;
        size_t _data_points_counter;
        std::map<std::string, SimpleTensor> _batch_history;
};


class AdaGrad : public Optimizer {
};


class AdaDelta : public Optimizer {
};


class Adam : public Optimizer {
};


#endif // __OPTIMIZER__