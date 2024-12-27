#ifndef __MODEL__
#define __MODEL__

#include "layer.h"
#include "utils.h"
#include "graph.h"

#include <map>
#include <string>
#include <cmath>


class Model {
    public:
        Model( std::vector<Layer*> layers ) {
            _graph_context = new Graph();
            for(Layer* layer : layers) {
                _layer_map.insert({layer->getName(), layer});
                std::cout << "layer: " << layer->getName() << " weight " << str_representation(layer->_weight.getSize()) << " bias " << str_representation(layer->_bias.getSize()) << "\n";
                layer->setGraph(_graph_context);
            }
            _layer_sequence = layers;
        }

        ~Model() {
            for(Layer* layer : _layer_sequence) {
                delete layer;
                layer = nullptr;
            }

            delete _graph_context;
            _graph_context = nullptr;
        }


        Tensor operator() (SimpleTensor input) {
            Tensor x;
            if(_graph_context != nullptr) {
                _graph_context -> clearSequence();
                x = Tensor(input, true, _graph_context);
            }
            else
                x = Tensor(input, false, nullptr);

            for(Layer* layer : _layer_sequence) {
                x = layer -> operator()(x);
            }
            return x;
        }

        void calcGrad(bool val=true) {
            if(val)
                _graph_context = new Graph();
            else
                _graph_context = nullptr;
    
            for(Layer* layer : _layer_sequence) 
                layer->setGraph(_graph_context);
        }

        Layer* operator[] (std::string layer_name) { return _layer_map[layer_name]; }

        std::vector<std::string> getLayers() {
            std::vector<std::string> keys;
            for(std::map<std::string, Layer*>::iterator it = _layer_map.begin(); it != _layer_map.end(); ++it) 
                keys.push_back(it->first);
            return keys;
        }

        SimpleTensor predict(SimpleTensor input) {
            Tensor x(input);
            for(Layer* layer : _layer_sequence) {
                x = layer -> operator()(x);
            }
            return x;
        }

        void setLossFun(std::string loss_name) {
            _loss_name = loss_name;
            _loss_fun = TensorOperations::loss(_loss_name);
        }

        Graph* getGraph() {
            return _graph_context;
        }

        std::map<std::string, float> gradient_report() {
            SimpleTensor input = SimpleTensor::rand({2, 1}, {0, 1});
            std::map<std::string, SimpleTensor> automatic_grad;
            std::map<std::string, SimpleTensor> manual_grad;

            this -> calcGrad(true);
            Tensor output = (*this)(input);
            _graph_context->backwards();

            for(Layer* layer : _layer_sequence) {
                std::string layer_name = layer -> getName(); 
                // std::cout << layer_name;
                automatic_grad.insert({ layer_name, _graph_context->getNode(_layer_map[layer_name]->getWeight())->getWholeGradValue() });
                // std::cout << _graph_context->getNode(_layer_map[layer_name]->getWeight())->getWholeGradValue()._all_elements;
            }
            // std::cout << "Gathered automatic\n";
        
            this -> calcGrad(false);
            for(Layer* layer : _layer_sequence) {
                // std::cout << "layer: " << layer->getName() << "\n";
                SimpleTensor weight = layer -> getWeight();
                manual_grad.insert({layer->getName(), SimpleTensor(weight.getSize(), 0.0)});
                for(int i = 0; i < weight.nElements(); i++) {
                    float org_value = weight._data[i];
                    float left_res, right_res;
                    float delta = 1e-3;

                    weight._data[i] -= delta;
                    left_res = this -> operator()(input)._data[0];
                    // std::cout << left_res << " ";
                    weight._data[i] += 2*delta;
                    right_res = this -> operator()(input)._data[0];
                    // std::cout << right_res << " ";
                    weight._data[i] = org_value;

                    manual_grad[layer->getName()]._data[i] = (right_res - left_res) / (2 * delta);
                    // std::cout << manual_grad[layer->getName()]._data[i] << " ";
                }
            }
            // std::cout << "Gathered manual\n";

            std::map<std::string, float> err_report;
            for(Layer* layer : _layer_sequence) {
                float avg_layer_error = 0;
                // std::cout << "auto: " << automatic_grad[layer->getName()];
                // std::cout << "auto: " << automatic_grad[layer->getName()]._all_elements;
                for(int i = 0; i < automatic_grad[layer->getName()]._all_elements; i++) {
                    avg_layer_error += std::pow(manual_grad[layer->getName()]._data[i] - automatic_grad[layer->getName()]._data[i], 2);
                }
                avg_layer_error /= automatic_grad[layer->getName()]._all_elements;
                err_report.insert({layer->getName(), avg_layer_error});
            }
            return err_report;
        }

    private:
        std::map<std::string, Layer*> _layer_map;
        std::vector<Layer*> _layer_sequence;
        std::string _loss_name;
        loss_fun _loss_fun;
        Graph* _graph_context;
};

#endif // __MODEL__