#include "tensor.h"
#include "utils.h"

#include <algorithm>
#include <cmath>

Tensor::Tensor(bool calc_grad, Graph* graph_context) :
    SimpleTensor(),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
        
    if(_calc_grad && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }

Tensor::Tensor(float value, bool calc_grad, Graph* graph_context) :
        Tensor(std::vector<size_t>{1},
        value, calc_grad,
        graph_context) { }

Tensor::Tensor(std::vector<size_t> size, float init, bool calc_grad, Graph* graph_context) :
    SimpleTensor(size, init),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
        
    if(_calc_grad  && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }

Tensor::Tensor(std::vector<size_t> size, std::vector<float> data, bool calc_grad, Graph* graph_context) :
    SimpleTensor(size, data),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
        
    if(_calc_grad  && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }

Tensor::Tensor(std::vector<size_t> size, float* data, bool calc_grad, Graph* graph_context) :
    SimpleTensor(size, data),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
        
    if(_calc_grad  && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }

Tensor::Tensor(SimpleTensor simple_tensor, bool calc_grad, Graph* graph_context) :
    SimpleTensor(simple_tensor),
    _calc_grad(calc_grad),
    _grad_graph(graph_context) {
        
    if(_calc_grad  && !_grad_graph->contains(*this) )
        _grad_graph -> addNode( new Node(*this) );
    }

Tensor::~Tensor() {
    // change to shhared pointer later
    // if(_slice == false)
    //     delete  []_data; // shared pointer
}

void Tensor::setGrapContext(Graph* graph_ptr) {
 _grad_graph = graph_ptr;
}

Graph* Tensor::getGraphContext() {
    return _grad_graph;
}

void Tensor::setCalcGrad(bool val) {
    _calc_grad = val;
}


// ############################## external functions ################################

// Tensor operator*(float sc, Tensor& t1) {
//     std::vector<size_t> s1 = t1._size;    
//     float* t2_data = new float[t1._all_elements];

//     for(int i = 0; i < t1._all_elements; i++)
//         t2_data[i] = t1._data[i] * sc;

//     return Tensor(s1, t2_data);
// }


Tensor TensorOperations::add(Tensor t1, Tensor t2) {
    if( t1._size != t2._size )
        std::cout << "(+) tensors size problem?!";

    // data of result tensor
    float* t3_data = new float[t1._all_elements];

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = t1._data[i] + t2._data[i];

    // adding node with local gradient to graph
    // For now lets assume only one of the tensors has graph attached
    // and its always left one. First attechment of graph need to be outside the Tensor
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( t1._calc_grad || t2._calc_grad ) {

        if( t1._grad_graph == nullptr || t2._grad_graph == nullptr )
            std::cout << "(+) no graph attached?!";

        if( t2._calc_grad ) // every tensor with _calc_grad should have the same graph context
            t2._grad_graph = t1._grad_graph;

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }

    SimpleTensor t3_simple = SimpleTensor(t1._size, t3_data);

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
    // cut out multipling and use operator* from simpletensor

    // tensors are restricted to dimensions t1, t2 -> max 2dims.
    // Derivative is not defined for higher dimensions. 
    std::vector<size_t> s1 = t1._size;
    std::vector<size_t> s2 = t2._size;

    if(s1.size() > 2 || s2.size() > 2)
        std::cout << "(+) too many dimensions to multiply correctly t1: " << s1.size() << "dims and t2: " << s1.size() << "dims ?!"; 

    // if t2 is vector then it should be made transformed [d, 1]
    
    // check for dim conditions
    size_t last_dim = s1[s1.size()-1];
    if( last_dim != s2[0]) // [a, b, ..., c, d] * [d, e] => [a, b, ..., c, e] 
        std::cout << "?";

    int number_of_all_rows = s1[0] * t1._cummulative_size[0] / last_dim; // a*b*...*c
    int row_length = last_dim;
    int t2_data_row_length = s2[ s2.size()-1 ];
    int number_of_columns = s2[1];

    // std::cout << "number_of_rows: " << number_of_all_rows << "\n";
    // std::cout << "row_length: " << row_length << "\n";
    // std::cout << "number_of_columns: " << number_of_columns << "\n";

    float* t1_data_row_ptr = t1._data;
    float* t3_data = new float[number_of_all_rows * number_of_columns]{0};

    float* t1_data_ptr = t1._data;
    float* t2_data_ptr = t2._data;
    float* t3_data_ptr = t3_data;
    
    for(int row = 0; row < number_of_all_rows; row++, t1_data_row_ptr += row_length) {
        for(int col = 0; col < number_of_columns; col++, t3_data_ptr++) {
            t1_data_ptr = t1_data_row_ptr; // for each column the same row must be iterated
            t2_data_ptr = t2._data + col; // iteration over column is taking element col from each row of matrix
            
            // t1 by elements in row. t2 by columns. row in t1 defines which element in each column t2
            for(int i = 0; i < row_length; i++, t2_data_ptr += t2_data_row_length, t1_data_ptr++)
                (*t3_data_ptr) += (*t1_data_ptr) * (*t2_data_ptr);
        }
    }

    std::vector<size_t> s3(s1);
    s3.pop_back();
    s3.push_back(s2[ s2.size()-1 ]);

    // adding node with local gradient to graph
    // For now lets assume only one of the tensors has graph attached
    // and its always left one. First attechment of graph need to be outside the Tensor
    bool t3_calc_grad = false;
    Graph* t3_graph_context = nullptr;
    if( t1._calc_grad || t2._calc_grad ) {

        if( t1._grad_graph == nullptr || t2._grad_graph == nullptr )
            std::cout << "(+) no graph attached?!";

        if( t2._calc_grad ) // every tensor with _calc_grad should have the same graph context
            t2._grad_graph = t1._grad_graph;

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }

    SimpleTensor t3_simple = SimpleTensor(s3, t3_data);

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
    }
    // std::cout << "done all\n";

    return Tensor(t3_simple, t3_calc_grad, t1._grad_graph);
}


// maybe all arguments should be simpletensor?
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
                *(der_t3byt1_data + + i*der_t3byt1_size[1]*der_t3byt1_size[2]*der_t3byt1_size[3] + j*der_t3byt1_size[2]*der_t3byt1_size[3] + k*der_t3byt1_size[3] + i) = t2.at({j, k});
    }

    // diag_len = std::min(der_t3byt2_size[0], der_t3byt2_size[ der_t3byt2_size.size()-1 ] )  ;
    for(size_t i = 0; i < der_t3byt2_size[0]; i++) {
        for(size_t j = 0; j < der_t3byt2_size[1]; j++)
            for(size_t l = 0; l < der_t3byt2_size[3]; l++)
                *(der_t3byt2_data + i*der_t3byt2_size[1]*der_t3byt2_size[2]*der_t3byt2_size[3] + j*der_t3byt2_size[2]*der_t3byt2_size[3] + j*der_t3byt2_size[3] + l) = t1.at({l, i});
    }

    // if there is n x 1 x 1 x m then it can be reduced to n x m. Because whatere will be multiplied the result will be n x 1 x 1 x k x ...
    // and because vector is always [k x 1] uneven dimensions don't exists then only if there is pair of 1, tensor can be reduced in dimensionality
    // removing pair of 1
    // if( std::count(der_t3byt1_size.begin(), der_t3byt1_size.end(), 1) == 2 )
    //     der_t3byt1_size.erase( std::remove(der_t3byt1_size.begin(), der_t3byt1_size.end(), 1), der_t3byt1_size.end() );
    // if( std::count(der_t3byt2_size.begin(), der_t3byt2_size.end(), 1) == 2 )
    //     der_t3byt2_size.erase( std::remove(der_t3byt2_size.begin(), der_t3byt2_size.end(), 1), der_t3byt2_size.end() );

    SimpleTensor der_t3byt1(der_t3byt1_size, der_t3byt1_data);
    SimpleTensor der_t3byt2(der_t3byt2_size, der_t3byt2_data);

    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({t1, der_t3byt1});
    derivatives.insert({t2, der_t3byt2});
    return derivatives;
}


Tensor TensorOperations::relu(Tensor t1) {
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
            std::cout << "(+) no graph attached?!";

        t3_calc_grad = true;
        t3_graph_context = t1._grad_graph;
    }

    SimpleTensor t3_simple = SimpleTensor(t1._size, t3_data);

    if(t3_calc_grad) {
        Node* t3_node = new Node(t3_simple);

        t3_node -> setOperation("relu");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(t1) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(t1));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = reluDerivatives(t1, t3_simple);
        t3_graph_context->getNode(t1) -> addLocalGradValue(t3_simple, derivatives[t1]);
    }

    return Tensor(t3_simple, t3_calc_grad, t1._grad_graph);
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
        std::cout << "(mse) argument can only be a scalar?!\n";
    
    if( p_size != r_size )
        std::cout << "(mse) arguments difrent sizes" << str_representation(p_size) << " vs " << str_representation(r_size) << " ?!\n";

    float* t3_data = new float[1]{ 0 };
    std::vector<size_t> t3_size{1, 1};
    t3_data[0] = std::pow(predicted._data[0] - real._data[0], 2);

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
        Node* t3_node = new Node(t3_simple);

        t3_node -> setOperation("mse");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(predicted) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(predicted));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = mseLossDerivatives(predicted, real, t3_simple);
        t3_graph_context->getNode(predicted) -> addLocalGradValue(t3_simple, derivatives[predicted]);
    }

    return Tensor(t3_simple, true, predicted._grad_graph);
}

Tensor TensorOperations::cceLoss(Tensor predicted, SimpleTensor real) {
    std::vector<size_t> p_size = predicted._size;
    std::vector<size_t> r_size = real._size;

    // the restriction for now is that CCE can be evaluated only on vector 
    if(p_size[1] != 1)
        std::cout << "(cce) argument can only be vector?!\n";
    
    if( p_size != r_size )
        std::cout << "(cce) arguments difrent sizes" << str_representation(p_size) << " vs " << str_representation(r_size) << " ?!\n";

    float* t3_data = new float[1]{ 0 };
    std::vector<size_t> t3_size{1, 1};

    size_t true_idx = 0;
    for(int i = 0; i < r_size[0]; i++) {
        if( real._data[i] == 1.0 )
            true_idx = i;
        (*t3_data) += std::exp( predicted._data[i] );
    }
    (*t3_data) = (-1) * std::log( predicted._data[true_idx] / t3_data[0] );

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
        Node* t3_node = new Node(t3_simple);

        t3_node -> setOperation("relu");
        t3_graph_context -> addNode( t3_node );
        t3_graph_context->getNode(predicted) -> addChild(t3_node);
        t3_graph_context->getNode(t3_simple) -> addParent(t3_graph_context->getNode(predicted));
        
        // adding derivatives
        std::map<std::string, SimpleTensor> derivatives = cceLossDerivatives(predicted, real, t3_simple);
        t3_graph_context->getNode(predicted) -> addLocalGradValue(t3_simple, derivatives[predicted]);
    }
    return Tensor(t3_simple, true, predicted._grad_graph);
}

std::map<std::string, SimpleTensor> TensorOperations::mseLossDerivatives(SimpleTensor predicted, SimpleTensor real, SimpleTensor t3) {
    std::vector<size_t> t3_size{1, 1};
    float* t3_data = new float[1];

    t3_data[0] = 2 * (predicted._data[0] - real._data[0]);

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

    // SimpleTensor der_t3byt1(der_t3byt1_size, der_t3byt1_data);
 
    std::map<std::string, SimpleTensor> derivatives;
    derivatives.insert({predicted, SimpleTensor()});
    return derivatives;
}


std::map<std::string, act_fun> TensorOperations::_activation_map {
    {"relu", &TensorOperations::relu}
};

std::map<std::string, loss_fun> TensorOperations::_loss_map {
    {"mse", &TensorOperations::mseLoss},
    {"cce", &TensorOperations::cceLoss}
};

        
std::ostream& operator<< (std::ostream& os, const Tensor& tensor) {
    std::vector<size_t> size = tensor.getSize();
    print_rec(os, tensor);
    os << "(tensor";
    print_size(os, tensor.getSize());
    os << ")";
    return os;
}
