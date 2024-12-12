#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#include <random>

#include "simple_tensor.h"
#include "utils.h"

SimpleTensor::SimpleTensor(): _slice{false}, _empty{true}, _all_elements{0} {
    _size = std::vector<size_t>{0};
}

SimpleTensor::SimpleTensor(float value): SimpleTensor(std::vector<size_t>{1}, value) { }


SimpleTensor::SimpleTensor(std::vector<size_t> size, float init) : _size{size}, _slice{false} {
    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;
    // std::cout << "construct: " << all_elements << "el\n";

    _data = new float[all_elements];
    std::fill_n(_data, all_elements, init);

    evaluateCummulatives();
    evaluateId();
}

SimpleTensor::SimpleTensor(std::vector<size_t> size, std::vector<float> data) : _size{size}, _slice{false} {
    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;

    // std::cout << "construct: " << all_elements << "el\n";

    if( data.size() != all_elements) {
        std::cout << "?";
    }

    _data = new float[all_elements];
    for(int i = 0; i < all_elements; i++)
        _data[i] = data[i];

    evaluateCummulatives();    
    evaluateId();
}

SimpleTensor::SimpleTensor(std::vector<size_t> size, float* data) : _size{size}, _data{data}, _slice{true} {
    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;

    evaluateCummulatives();    
    evaluateId();
}


SimpleTensor::~SimpleTensor() {
    // change to shhared pointer later
    // if(_slice == false)
    //     delete  []_data; // shared pointer
}


SimpleTensor SimpleTensor::identity(size_t size) {
    SimpleTensor tens = SimpleTensor({size, size}, 0.0);
    for(int i = 0; i < size; i++)
        tens._data[i + i * size] = 1;
    return tens;
}


SimpleTensor SimpleTensor::rand(std::vector<size_t> size, std::pair<float, float> range) {
    SimpleTensor tens = SimpleTensor({size}, 0.0);
    
    // this need to be placed as some global value
    std::mt19937 rand_engine;
    rand_engine.seed(101);

    std::uniform_real_distribution distribution(0.0, 1.0);

    for(int i = 0; i < tens._all_elements; i++)
        tens._data[i] = distribution(rand_engine);

    return tens;
}

void SimpleTensor::evaluateCummulatives() {
    _cummulative_size = std::vector<size_t>(_size.size());
    for(int i = 0; i < _size.size(); i++) {
        size_t cummulation = 1;
        for(int j = i+1; j < _size.size(); j++)
            cummulation *= _size[j];
        _cummulative_size[i] = cummulation;
    }
}

void SimpleTensor::evaluateId() {
    std::stringstream ss;
    ss << _data;
    std::string id = ss.str();
    id += '#';

    for(auto el : _size)
        id += std::to_string(el);
    
    _id = id;
}

std::vector<size_t> SimpleTensor::getSize() const { return _size; };


void SimpleTensor::trim() {
    int len = _size.size();
    if( len > 2 && _size[len-1] == 1 && _size[len-2] == 1 ){
        _size.pop_back();
        _size.pop_back();
    }
}


SimpleTensor SimpleTensor::operator[] (size_t idx) {
    float *slice_start{nullptr};
    std::vector<size_t> size{0};

    if(idx < _size[0]) {
        size = _size;
        size.erase(size.begin());

        int elements_per_slice{1};
        for(int i = 0; i < size.size(); i++)
            elements_per_slice *= size[i];
        slice_start = _data + idx * elements_per_slice;

    } else {
        std::cout << "?";
    }

    return SimpleTensor(size, slice_start);
}

const SimpleTensor SimpleTensor::operator[] (size_t idx) const {
    float *slice_start{nullptr};
    std::vector<size_t> size{0};

    if(idx < _size[0]) {
        size = _size;
        size.erase(size.begin());

        int elements_per_slice{1};
        for(int i = 0; i < size.size(); i++)
            elements_per_slice *= size[i];
        slice_start = _data + idx * elements_per_slice;

    } else {
        std::cout << "?";
    }

    return SimpleTensor(size, slice_start);
}

float SimpleTensor::at(std::vector<size_t> point) const {
    // add some size table instead of calculating it every time
    // strats with last value
    int flat_idx = 0;
    for(int i = 0; i < _size.size(); i++)
        flat_idx += point[i] * _cummulative_size[ i ];

    return *(_data + flat_idx);
}


SimpleTensor SimpleTensor::operator+= (SimpleTensor t1) {

    if(_empty) {
        _size = t1._size;
        _all_elements = t1._all_elements;
        _data = new float[t1._all_elements]{ 0 } ;
    }

    if( t1._size != _size )
        std::cout << "(+=) dimensions do not match: " << str_representation(_size) << " vs " << str_representation(t1._size) << "?!\n";

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        _data[i] += t1._data[i];

    return (*this);
}



SimpleTensor operator+(SimpleTensor t1, SimpleTensor t2) {
    if( t1._size != t2._size )
        std::cout << "(+) dimensions do not match: " << str_representation(t1._size) << " vs " << str_representation(t2._size) << "?!\n";

    // data of result tensor
    float* t3_data = new float[t1._all_elements];

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = t1._data[i] + t2._data[i];

    return SimpleTensor(t1._size, t3_data);
}

SimpleTensor operator*(float sc, SimpleTensor t1) {
    // std::cout << "scalar * tensor\n";

    // data of result tensor
    float* t3_data = new float[t1._all_elements];

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = sc * t1._data[i] ;

    return SimpleTensor(t1._size, t3_data);
}


SimpleTensor operator*(SimpleTensor t1, SimpleTensor t2) {
    std::vector<size_t> s1 = t1._size;
    std::vector<size_t> s2 = t2._size;

    // if t2 is vector then it should be made transformed [d, 1]
    
    // check for dim conditions
    size_t last_dim = s1[s1.size()-1];
    if( last_dim != s2[0]) // [a, b, ..., c, d] * [d, e] => [a, b, ..., c, e] 
        std::cout << "(*) dimensions do not match: " << str_representation(s1) << " vs " << str_representation(s2) << "?!\n";

    int number_of_all_rows = s1[0] * t1._cummulative_size[0] / last_dim; // a*b*...*c
    int row_length = last_dim;
    int t2_data_row_length = s2[ s2.size()-1 ];
    int number_of_columns = s2[1];

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

    return SimpleTensor(s3, t3_data);
}





void print_rec(std::ostream& os, const SimpleTensor tensor, int depth) {
    std::vector<size_t> size( tensor.getSize() );

    if( size.size() == 1 ) {
        for(int k = 0; k < depth; k++)       
            os << " ";
        os << "[ ";
        for(unsigned int i = 0; i < size[0]; i++)
            os << tensor.at({i}) << " ";
        os << "]\n";
    }
    
    else {
        for(int k = 0; k < depth; k++)       
            os << " ";
        os << "[\n";
        for(int j = 0; j < size[0]; j++)
            print_rec(os, tensor[j], depth+1);
        for(int k = 0; k < depth; k++)       
            os << " ";
        os << "]\n";
    }
}

void print_size(std::ostream& os, const std::vector<size_t> vec) {
    os << "[" << vec[0];
    for(int i = 1; i < vec.size(); i++)
        os << ", " << vec[i];
    os << "]";
}

std::ostream& operator<< (std::ostream& os, const SimpleTensor& tensor) {
    std::vector<size_t> size = tensor.getSize();
    print_rec(os, tensor);
    os << "(tensor";
    print_size(os, tensor.getSize());
    os << ")";
    return os;
}

