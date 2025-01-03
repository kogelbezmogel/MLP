#ifndef __MODEL__
#define __MODEL__

#include "layer.h"
#include "utils.h"
#include "graph.h"

#include <map>
#include <string>
#include <cmath>


class Model {

    friend std::ostream& operator<<(std::ostream& os, const Model& model);

    public:
        Model( std::vector<Layer*> layers );

        ~Model();

        Tensor operator() (SimpleTensor input);

        void calcGrad(bool val=true);

        Layer* operator[] (std::string layer_name);

        std::vector<std::string> getLayers();

        SimpleTensor predict(SimpleTensor input);

        void setLossFun(std::string loss_name);

        Graph* getGraph();

        std::map<std::string, float> gradient_report();

        Layer* getLayer(std::string name);

    private:
        std::map<std::string, Layer*> _layer_map;
        std::vector<Layer*> _layer_sequence;
        std::string _loss_name;
        loss_fun _loss_fun;
        Graph* _graph_context;
};

#endif // __MODEL__