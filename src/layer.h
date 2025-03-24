#ifndef __LAYER__
#define __LAYER__


#include "tensor.h"

class Layer {

    friend class Model;

    public:
        Layer(size_t in_features, size_t out_features, bool bias=true, std::string activation="");
        Tensor operator() (Tensor input); 

        void setGraph(Graph* graph);

        std::string getName();

        Tensor& getWeight();

        Tensor& getBias();

        bool bias() const;

        void setWeight( SimpleTensor weight );

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

#endif //__LAYER__