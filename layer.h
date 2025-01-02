#ifndef __LAYER__
#define __LAYER__


#include "tensor.h"

class Layer {

    friend class Model;

    public:
        Layer(size_t in_features, size_t out_features, bool bias=true, std::string activation="") : 
            _in_features(in_features),
            _out_features(out_features),
            _has_bias(bias),
            _activation(activation) {

                _weight = SimpleTensor::rand({_out_features, _in_features}, {0, 1});
                if(_has_bias)
                    _bias = SimpleTensor::rand({_out_features, 1}, {0, 1});

                _layer_name = "layer_" + std::to_string(_layer_count);
                _layer_count++;
        }

        Tensor operator() (Tensor input) {
            // std::cout << _layer_name << " operator()\n";
            if(_has_bias && _activation != "")
                return TensorOperations::activation(_activation)( TensorOperations::add( TensorOperations::mul(_weight, input), _bias ) );
            else if(!_has_bias && _activation == "")
                return TensorOperations::mul(_weight, input);
            else if(_has_bias && _activation == "")
                return TensorOperations::add( TensorOperations::mul(_weight, input), _bias );
            else
                return TensorOperations::activation(_activation)( TensorOperations::mul(_weight, input));
        }

        void setGraph(Graph* graph) {
            if(graph != nullptr) {
                _weight.setCalcGrad(true);
                graph -> addNode( new Node(_weight, false) );
            } else
                _weight.setCalcGrad(false);
            _weight.setGrapContext(graph);
            

            if(_has_bias) {
                if(graph != nullptr) {
                    _bias.setCalcGrad(true);
                    graph -> addNode( new Node(_bias, false) );
                } else
                    _bias.setCalcGrad(false);                
                _bias.setGrapContext(graph);
            }
        }

        // void update(SimpleTensor delta_weight);
        
        std::string getName() { return _layer_name; }

        Tensor& getWeight() { return _weight; }

        Tensor& getBias() { return _bias; }

        bool bias() const { return _has_bias; }

        void setWeight( SimpleTensor weight ) {
            for(size_t i = 0; i < _weight.getSize()[0]; i++)
                for(size_t j = 0; j < _weight.getSize()[1]; j++)
                    _weight.set(weight.at({i, j}), {i, j});
        }

    private:
        static int _layer_count;

        size_t _in_features;
        size_t _out_features;

        Graph* _model_graph;

        Tensor _weight;
        Tensor _bias;
        std::string _activation;

        bool _has_bias;
        std::string _layer_name;
};

// to cpp file
int Layer::_layer_count = 0;

#endif //__LAYER__