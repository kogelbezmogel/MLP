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
        

// Tensor::Tensor(bool calc_grad, Graph* graph_context) :
//     SimpleTensor(),
//     _calc_grad(calc_grad),
//     _grad_graph(graph_context) {
        
//     if(_calc_grad && !_grad_graph->contains(*this) )
//         _grad_graph -> addNode( new Node(*this) );
//     }

// Tensor::Tensor(float value, bool calc_grad, Graph* graph_context) :
//         Tensor(std::vector<size_t>{1},
//         value, calc_grad,
//         graph_context) { }


// Tensor::Tensor(Tensor&& to_move) : SimpleTensor(std::move(to_move)) {
//     std::cout << "moved\n";
//     _calc_grad = to_move._calc_grad;
//     _grad_graph = to_move._grad_graph;

//     to_move._calc_grad = false;
//     to_move._grad_graph = nullptr;
// }


// Tensor::Tensor(std::vector<size_t> size, float* data, bool calc_grad, Graph* graph_context) :
//     SimpleTensor(size, data),
//     _calc_grad(calc_grad),
//     _grad_graph(graph_context) {
        
//     if(_calc_grad  && !_grad_graph->contains(*this) )
//         _grad_graph -> addNode( new Node(*this) );
//     }






// ############################## external functions ################################



Tensor TensorOperations::add(Tensor t1, Tensor t2) {
    SimpleTensor t3_simple(t1 + t2);

    // adding node with local gradient to graph
    // For now lets assume only one of the tensors has graph attached
    // and its always left one. First attechment of graph need to be outside the Tensor
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( t1._calc_grad || t2._calc_grad ) {

        if( t1._grad_graph == nullptr || t2._grad_graph == nullptr )
            throw NoGraphAttachedException("No graph attached.\n");

        if( t2._calc_grad ) // every tensor with _calc_grad should have the same graph context
            t2._grad_graph = t1._grad_graph;

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple);

        t3_node -> setOperation("add");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(t1) -> addChild(t3_node);
        t3_graph_context->getNode(t2) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t1));
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t2));

        // adding derivatives
        t3_graph_context->getNode(t1) -> addLocalGradValue(t3_simple, SimpleTensor::identity( t1.getSize()[0] ));
        t3_graph_context->getNode(t2) -> addLocalGradValue(t3_simple, SimpleTensor::identity( t2.getSize()[0] ));
    }
    return Tensor(t3_simple, t3_calc_grad, t1._grad_graph);
}


Tensor TensorOperations::mul(Tensor t1, Tensor t2) {
    // std::cout << "fun: mul\n";
    SimpleTensor t3_simple = t1 * t2;
    
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( t1._calc_grad || t2._calc_grad ) {

        if( t1._grad_graph == nullptr || t2._grad_graph == nullptr )
            throw NoGraphAttachedException("No graph attached.\n");

        if( t2._calc_grad ) // every tensor with _calc_grad should have the same graph context
            t2._grad_graph = t1._grad_graph;

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }
    // std::cout << "  - calculated\n";



    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple);
        t3_node -> setOperation("mul");

        t3_graph_context -> addNode( t3_node );

        t3_graph_context->getNode(t1) -> addChild(t3_node);
        t3_graph_context->getNode(t2) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t1));
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t2));

        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = mulDerivatives(t1, t2, t3_simple);

        t3_graph_context->getNode(t1) -> addLocalGradValue(t3_simple, derivatives[t1]);
        t3_graph_context->getNode(t2) -> addLocalGradValue(t3_simple, derivatives[t2]);
        // std::cout << "  - added to graph\n";


        // std::cout << "Adding " << str_representation() << 
    }
    // std::cout << "done all\n";

    // move constructor makes t3_simple empty because of move
    // std::cout << "fun: mul end\n";
    return Tensor(t3_simple, t3_calc_grad, t1._grad_graph);
}


// // maybe all arguments should be simpletensor?
std::map<std::string, SimpleTensor> TensorOperations::mulDerivatives(SimpleTensor t1, SimpleTensor t2, SimpleTensor t3) {
    std::vector<size_t> s1 = t1._size;
    std::vector<size_t> s2 = t2._size;
    std::vector<size_t> s3 = t3._size;

    float* der_t3byt1_data = new float[t3._all_elements * t1._all_elements]{ 0 };
    float* der_t3byt2_data = new float[t3._all_elements * t2._all_elements]{ 0 };

    std::vector<size_t> der_t3byt1_size;
    der_t3byt1_size.insert( der_t3byt1_size.end(), s1.begin(), s1.end() );
    der_t3byt1_size.insert( der_t3byt1_size.end(), s3.begin(), s3.end()-2 );
    der_t3byt1_size.insert( der_t3byt1_size.end(), s3.rbegin(), s3.rbegin()+2 );
    
    std::vector<size_t> der_t3byt2_size;
    der_t3byt2_size.insert( der_t3byt2_size.end(), s2.begin(), s2.end() );
    der_t3byt2_size.insert( der_t3byt2_size.end(), s3.begin(), s3.end()-2 );
    der_t3byt2_size.insert( der_t3byt2_size.end(), s3.rbegin(), s3.rbegin()+2 );

    // this can be in separeted subfunction
    // int diag_len = std::min(der_t3byt1_size[0], der_t3byt1_size[ der_t3byt1_size.size()-1 ] )  ;
    for(size_t i = 0; i < der_t3byt1_size[0]; i++) {
        for(size_t j = 0; j < der_t3byt1_size[1]; j++)
            for(size_t k = 0; k < der_t3byt1_size[2]; k++)
                *(der_t3byt1_data + i*der_t3byt1_size[1]*der_t3byt1_size[2]*der_t3byt1_size[3] + j*der_t3byt1_size[2]*der_t3byt1_size[3] + k*der_t3byt1_size[3] + i) = t2.at({j, k});
    }

    // diag_len = std::min(der_t3byt2_size[0], der_t3byt2_size[ der_t3byt2_size.size()-1 ] )  ;
    for(size_t i = 0; i < der_t3byt2_size[0]; i++) {
        for(size_t j = 0; j < der_t3byt2_size[1]; j++)
            for(size_t l = 0; l < der_t3byt2_size[3]; l++)
                *(der_t3byt2_data + i*der_t3byt2_size[1]*der_t3byt2_size[2]*der_t3byt2_size[3] + j*der_t3byt2_size[2]*der_t3byt2_size[3] + j*der_t3byt2_size[3] + l) = t1.at({l, i});
    }

    SimpleTensor der_t3byt1(der_t3byt1_size, der_t3byt1_data);
    SimpleTensor der_t3byt2(der_t3byt2_size, der_t3byt2_data);

    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({t1, der_t3byt1});
    derivatives.insert({t2, der_t3byt2});
    return derivatives;
}


Tensor TensorOperations::relu(Tensor t1) {
    // std::cout << "fun: relu\n";

    // data of result tensor
    float* t3_data = new float[t1._all_elements];

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = std::max(t1._data[i], 0.0f);

    // adding node with local gradient to graph
    // For now lets assume only one of the tensors has graph attached
    // and its always left one. First attechment of graph need to be outside the Tensor
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( t1._calc_grad ) {
        if( t1._grad_graph == nullptr )
            throw NoGraphAttachedException("No graph attached.\n");

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }

    SimpleTensor t3_simple(t1._size, t3_data);

    // std::cout << "  - calculated\n";

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple); // !!!!

        t3_node -> setOperation("relu");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(t1) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t1));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = reluDerivatives(t1, t3_simple);
        t3_graph_context->getNode(t1) -> addLocalGradValue(t3_simple, derivatives[t1]);
        // std::cout << "  - added to graph\n";

    }

    // std::cout << "fun: relu end\n";
    return Tensor(t3_simple, t3_calc_grad, t3_graph_context);
}

std::map<std::string, SimpleTensor> TensorOperations::reluDerivatives(SimpleTensor t1, SimpleTensor t3) {
    std::vector<size_t> s1 = t1._size;
    std::vector<size_t> s3 = t3._size;

    float* der_t3byt1_data = new float[t3._all_elements * t1._all_elements]{ 0 };

    std::vector<size_t> der_t3byt1_size;
    der_t3byt1_size.insert( der_t3byt1_size.end(), s1.begin(), s1.end() );
    der_t3byt1_size.insert( der_t3byt1_size.end(), s3.begin(), s3.end()-2 );
    der_t3byt1_size.insert( der_t3byt1_size.end(), s3.rbegin(), s3.rbegin()+2 );
    
    // this can be in separeted subfunction
    for(size_t i = 0; i < der_t3byt1_size[0]; i++) {
        for(size_t j = 0; j < der_t3byt1_size[1]; j++)
                *(der_t3byt1_data + + i*der_t3byt1_size[1]*der_t3byt1_size[2]*der_t3byt1_size[3] + j*der_t3byt1_size[2]*der_t3byt1_size[3] + j*der_t3byt1_size[3] + i) = (t1.at({j, i}) > 0);
    }

    SimpleTensor der_t3byt1(der_t3byt1_size, der_t3byt1_data);
    
    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({t1, der_t3byt1});
    return derivatives;
}


Tensor TensorOperations::mseLoss(Tensor predicted, SimpleTensor real) {
    std::vector<size_t> p_size = predicted._size;
    std::vector<size_t> r_size = real._size;

    // the restriction for now is that CCE can be evaluated only on vector 
    if(p_size[0] != 1 || p_size[1] != 1)
        throw MustBeScalarException("(MSE) loss. Argument can only be a scalar\n");

    if( p_size != r_size )
        throw WrongDimensionsException( "(MSE) loss. Arguments with difrent sizes y_pred: " +  str_representation(p_size) + " y_real: " + str_representation(r_size) );

    float* t3_data = new float[1]{ 0 };
    std::vector<size_t> t3_size{1, 1};
    t3_data[0] = std::pow(predicted._data[0] - real._data[0], 2);

    // graph 
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( predicted._calc_grad ) {

        if( predicted._grad_graph == nullptr )
            throw NoGraphAttachedException("No graph attached.\n");

        t3_calc_grad = true;
        t3_graph_context = predicted._grad_graph;
    }

    SimpleTensor t3_simple(t3_size, t3_data);

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple); // !!!!

        t3_node -> setOperation("mse");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(predicted) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(predicted));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = mseLossDerivatives(predicted, real, t3_simple);
        t3_graph_context->getNode(predicted) -> addLocalGradValue(t3_simple, derivatives[predicted]);
    }

    Tensor t(t3_simple, true, predicted._grad_graph);
    return t;
}


Tensor TensorOperations::cceLoss(Tensor predicted, SimpleTensor real) {
    // real need to be a label of true class therefore a index at which the biggest value should appear. size [1 x 1]
    // predicted it the same time would be a vector of size [n x 1]

    std::vector<size_t> p_size = predicted._size;
    std::vector<size_t> r_size = real._size;

    // the restriction for now is that CCE can be evaluated only on vector 
    if(p_size[0] != 1 && p_size[1] != 1)
        std::cout << "(cce) argument can only be a scalar value?!\n";
    
    float* t3_data = new float[1]{ 0 };
    std::vector<size_t> t3_size{1, 1};

    // size_t true_idx = 0;
    size_t true_idx = real.at({0, 0});

    for(int i = 0; i < p_size[0]; i++) {
        // if( real._data[i] == 1.0 )
        //     true_idx = i;
        t3_data[0] += std::exp( predicted._data[i] );
    }

    t3_data[0] = (-1) * std::log( std::exp(predicted._data[true_idx]) / t3_data[0] );
    // t3_data[0] = std::exp(predicted._data[true_idx]) / t3_data[0];

    // graph 
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( predicted._calc_grad ) {

        if( predicted._grad_graph == nullptr )
            std::cout << "(+) no graph attached?!";

        t3_calc_grad = true;
        t3_graph_context = predicted._grad_graph;
    }

    SimpleTensor t3_simple = SimpleTensor(t3_size, t3_data);

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple); // !!!!

        t3_node -> setOperation("cce");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(predicted) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(predicted));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = cceLossDerivatives(predicted, real, t3_simple);
        t3_graph_context->getNode(predicted) -> addLocalGradValue(t3_simple, derivatives[predicted]);
    }
    return Tensor(t3_simple, true, predicted._grad_graph);
}


Tensor TensorOperations::bceLoss(Tensor predicted, SimpleTensor real) {
    // here always predicted and real are the same size [1 x 1] real is 0 or 1 and predicted is number from range [0, 1].
    // function takes predicted as a real number from range (-inf, +inf). To omit problems it constrained to (-100 +100)
    // Probablitity means the probability of label=1 not 0.

    std::vector<size_t> p_size = predicted._size;
    std::vector<size_t> r_size = real._size;

    // add check for dimenions and sizes !!

    float* t3_data = new float[1]{ 0 };
    std::vector<size_t> t3_size{1, 1};

    size_t label = real.at({0, 0});

    // contrain check
    if( predicted._data[0] < -100 )
        predicted._data[0] == -100.0;
    else if( predicted._data[0] > 100)
        predicted._data[0] == 100.0;

    // calculating probability of true (sigmoid)
    t3_data[0] = 1.0 / (1 + std::exp(-predicted._data[0]));

    // using probaility to get bce(y, y_t)
    if( label == 1 ) {
        t3_data[0] = -std::log(t3_data[0]);
    } else if( label == 0 ) {
        t3_data[0] = -std::log(1 - t3_data[0]);
    } else {
        // throw some exception !!!
    }

    // graph 
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( predicted._calc_grad ) {

        if( predicted._grad_graph == nullptr )
            std::cout << "(+) no graph attached?!"; // !!! add exception

        t3_calc_grad = true;
        t3_graph_context = predicted._grad_graph;
    }

    SimpleTensor t3_simple = SimpleTensor(t3_size, t3_data);

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple);

        t3_node -> setOperation("bce");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(predicted) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(predicted));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = bceLossDerivatives(predicted, real, t3_simple);
        t3_graph_context->getNode(predicted) -> addLocalGradValue(t3_simple, derivatives[predicted]);
    }
    return Tensor(t3_simple, true, predicted._grad_graph);
}


std::map<std::string, SimpleTensor> TensorOperations::mseLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3) {
    std::vector<size_t> t3_size{1, 1};
    float* t3_data = new float[1];

    t3_data[0] = 2 * (predicted._data[0] - real._data[0]); // this can be chabges to do w = w + lr*dw  instead of w = w - lr*dw

    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({predicted, SimpleTensor(t3_size, t3_data)});
    return derivatives;
}


std::map<std::string, SimpleTensor> TensorOperations::cceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3) {
    // std::vector<float> res(vec.size(), 0);
    // for(int i=0; i<vec.size(); i++) {
    //     float x=0, y=0, sec=0;

    //     x = std::exp( vec[i] );
    //     for(float el : vec)
    //         y += std::exp(el);

    //     if(i==y_true)
    //         sec = 1;
    //     res[i] = x / y - sec;
    // }

    std::vector<size_t> der_t3bypredicted_size = predicted.getSize();
    // std::reverse(der_t3bypredicted_size.begin(), der_t3bypredicted_size.end()); // derivative is of size of transposed predicted tensor
    float* der_t3bypredicted_data = new float[predicted._all_elements];

    // summing exponent of all element in predicted vector
    float general_sum = 0;
    for(int i = 0; i < predicted._all_elements; i++)
        general_sum += std::exp(predicted._data[i]);    

    // creating fractions with general sum
    for(int i = 0; i < predicted._all_elements; i++)
        der_t3bypredicted_data[i] = std::exp(predicted._data[i]) / general_sum;
    der_t3bypredicted_data[ (size_t) real._data[0] ] -= 1; // for true_index derivative is (fraction - 1)

    SimpleTensor der_t3bypredicted(der_t3bypredicted_size, der_t3bypredicted_data);
 
    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({predicted, der_t3bypredicted});
    return derivatives;
}


std::map<std::string, SimpleTensor> TensorOperations::bceLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3) {
    std::vector<size_t> der_t3bypredicted_size = predicted.getSize();
    // std::reverse(der_t3bypredicted_size.begin(), der_t3bypredicted_size.end()); // derivative is of size of transposed predicted tensor
    float* der_t3bypredicted_data = new float[predicted._all_elements];

    float probability = t3._data[0];

    size_t label = real._data[0];
    
    // creating fractions with general sum
    if(label == 1) {
        der_t3bypredicted_data[0] = probability;
    }
    else if(label == 0) {
        der_t3bypredicted_data[0] = 1 - probability;
    } else {
        // throw some exception
    }
    SimpleTensor der_t3bypredicted(der_t3bypredicted_size, der_t3bypredicted_data);
 
    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({predicted, der_t3bypredicted});
    return derivatives;
}


std::map<std::string, act_fun> TensorOperations::_activation_map {
    {"relu", &TensorOperations::relu}
};


std::map<std::string, loss_fun> TensorOperations::_loss_map {
    {"mse", &TensorOperations::mseLoss},
    {"cce", &TensorOperations::cceLoss},
    {"bce", &TensorOperations::bceLoss}
};

        
// std::ostream& operator<< (std::ostream& os, const Tensor& tensor) {
//     std::vector<size_t> size = tensor.getSize();
//     print_rec(os, tensor);
//     os << "(tensor";
//     print_size(os, tensor.getSize());
//     os << ")";
//     return os;
// }
