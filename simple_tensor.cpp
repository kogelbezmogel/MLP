#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <random>

#include "simple_tensor.h"
#include "utils.h"
#include "except.h"

// initialization of random engine
std::mt19937 SimpleTensor::rand_engine { (long unsigned int) time(NULL) % 101 };


SimpleTensor::SimpleTensor(): _slice{false}, _empty{true}, _all_elements{0} {
    // std::cout << __PRETTY_FUNCTION__ << "\n";    
    _id = "_empty";
    _size = std::vector<size_t>{0};
    _strong_ref_count = nullptr;
    // _weak_ref_count = nullptr;
}


SimpleTensor::SimpleTensor(std::vector<size_t> size, float init) : _size{size}, _slice{false} {
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << "\n";
    // _weak_ref_count = new size_t{0};
    _strong_ref_count = new size_t{1};

    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;

    _data = new float[all_elements];
    std::fill_n(_data, all_elements, init);

    evaluateCummulatives();
    evaluateId();
}


SimpleTensor::SimpleTensor(const SimpleTensor& to_copy) {
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << "\n";
    _cummulative_size = to_copy._cummulative_size;
    _size = to_copy._size;
    _empty = to_copy._empty;
    _all_elements = to_copy._all_elements;
    _slice = to_copy._slice;  
    _id = to_copy._id;
    _data = to_copy._data;

    _strong_ref_count = to_copy._strong_ref_count;
    if(_strong_ref_count && !_slice)
        (*_strong_ref_count)++;

    // _weak_ref_count = to_copy._weak_ref_count;
    // else if (_slice)
        // (*_weak_ref_count)++;
}


SimpleTensor::SimpleTensor(std::vector<size_t> size, std::vector<float> data) : _size{size}, _slice{false} {
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << "\n";
    // _weak_ref_count = new size_t{0};
    _strong_ref_count = new size_t{1};

    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;

    if( data.size() != all_elements) {
        throw WrongSizeException(
            "Source vector size doesn't match. Trying to fill tensor of "
            + std::to_string(all_elements)
            + " elements with data of length "
            + std::to_string(data.size())
        );
    }

    _data = new float[all_elements];
    for(int i = 0; i < all_elements; i++)
        _data[i] = data[i];

    evaluateCummulatives();    
    evaluateId();
}


SimpleTensor::SimpleTensor(std::vector<size_t> size, float* data, bool slice) : _size{size}, _data{data}, _slice{slice} {
    // std::cout << __PRETTY_FUNCTION__ << "\n";
    // _weak_ref_count = new size_t{0};
    _strong_ref_count = new size_t{1};

    size_t all_elements = 1;
    for(size_t k : size)
        all_elements *= k;
    _all_elements = all_elements;

    evaluateCummulatives();    
    evaluateId();
}


SimpleTensor::SimpleTensor(SimpleTensor&& to_move) {
    // std::cout << __PRETTY_FUNCTION__ << "\n";
    // _weak_ref_count = to_move._weak_ref_count;
    _strong_ref_count = to_move._strong_ref_count;
    _cummulative_size = std::move(to_move._cummulative_size);
    _size = std::move(to_move._size);
    _empty = to_move._empty;
    _id = to_move._id;
    _all_elements = to_move._all_elements;
    _data = to_move._data;
    _slice = to_move._slice;  

    // make to_move empty
    // to_move._weak_ref_count = nullptr;
    to_move._empty = true;
    to_move._strong_ref_count = nullptr;
    to_move._cummulative_size.clear();
    to_move._size.clear();
    to_move._id = "";
    to_move._all_elements = 0;
    to_move._data = nullptr;
    to_move._slice = false;
}


SimpleTensor::~SimpleTensor() {
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this);
    __clean_up__();
}


void SimpleTensor::__clean_up__() {
    // std::cout << __PRETTY_FUNCTION__ << "\n";

    if(_empty) {
        //nothing to do
    } else if(_slice == false) {
        (*_strong_ref_count)--;
        // std::cout << " s_ref:" << *_strong_ref_count << "\n";

        if(*_strong_ref_count == 0) {
            // if(*_weak_ref_count != 0) {
                // std::cout << *_weak_ref_count << " weak refs stiil around?! ";
            // }
            // std::cout << " memory deallocated ";
            delete _strong_ref_count;
            // delete _weak_ref_count;
            delete [] _data;

            _strong_ref_count = nullptr;
            // _weak_ref_count = nullptr;
            _data = nullptr;
        }
    } else if (_slice == true ) {
        // std::cout << " w_ref: " << *_weak_ref_count << "\n";
        // (*_weak_ref_count)--;
    }
}


SimpleTensor& SimpleTensor::operator=(const SimpleTensor& to_copy) {    
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << "\n";
    __clean_up__();

    _data = to_copy._data;
    _cummulative_size = to_copy._cummulative_size;
    _size = to_copy._size;
    _empty = to_copy._empty;
    _all_elements = to_copy._all_elements;
    _slice = to_copy._slice;  
    _id = to_copy._id;

    _strong_ref_count = to_copy._strong_ref_count;

    if(_strong_ref_count && !_slice)
        (*_strong_ref_count)++;

    // _weak_ref_count = to_copy._weak_ref_count;
    // else if (_slice)
        // (*_weak_ref_count)++;
    return (*this);
}


SimpleTensor& SimpleTensor::operator=(SimpleTensor && to_move) {
    // std::cout << __PRETTY_FUNCTION__ << "\n";

    // freeing old memory
    __clean_up__();

    // moving
    _cummulative_size = std::move(to_move._cummulative_size);
    _size = std::move(to_move._size);
    _empty = to_move._empty;
    _id = to_move._id;
    _all_elements = to_move._all_elements;
    _data = to_move._data;
    _slice = to_move._slice;  
    _strong_ref_count = to_move._strong_ref_count;

    // make to_move empty
    to_move._empty = true;
    to_move._slice = false;
    to_move._cummulative_size.clear();
    to_move._size.clear();
    to_move._id = "_empty";
    to_move._all_elements = 0;
    to_move._data = nullptr;
    to_move._slice = false;
    to_move._strong_ref_count = nullptr;

    return (*this);
}


SimpleTensor SimpleTensor::operator[] (size_t idx) {
    SimpleTensor t1;
    float *slice_start{nullptr};
    std::vector<size_t> size{0};

    if(idx < _size[0]) {
        size = _size;
        size.erase(size.begin());

        int elements_per_slice{1};
        for(int i = 0; i < size.size(); i++)
            elements_per_slice *= size[i];
        slice_start = _data + idx * elements_per_slice;

        // some kind of constructor would be better
        t1._size = size;
        t1._data = slice_start;
        t1._id = _id + "_s";
        t1._empty = false;
        t1._slice = true;
        t1._all_elements = elements_per_slice;
        t1._strong_ref_count = _strong_ref_count;
        // t1._weak_ref_count = _weak_ref_count;
        // (*_weak_ref_count)++;
        t1.evaluateCummulatives();
    } else {
        throw IndexOutOfRangeException(
            "Index "
            + std::to_string(idx)
            + " out of range for tensor of shape: "
            + str_representation(_size)
        );
    }
    return t1;
}


const SimpleTensor SimpleTensor::operator[] (size_t idx) const {
    SimpleTensor t1;
    float *slice_start{nullptr};
    std::vector<size_t> size{0};

    if(idx < _size[0]) {
        size = _size;
        size.erase(size.begin());

        int elements_per_slice{1};
        for(int i = 0; i < size.size(); i++)
            elements_per_slice *= size[i];
        slice_start = _data + idx * elements_per_slice;

        // some kind of constructor would be better
        t1._size = size;
        t1._data = slice_start;
        t1._id = _id + "_s";
        t1._empty = false;
        t1._slice = true;
        t1._all_elements = elements_per_slice;
        t1._strong_ref_count = _strong_ref_count;
        // t1._weak_ref_count = _weak_ref_count;
        // (*_weak_ref_count)++;
        t1.evaluateCummulatives();
    } else {
        std::cout << "Index " << idx << " out of range?!\n";
    }
    // std::cout << __PRETTY_FUNCTION__ << "\n";
    return t1;
}


SimpleTensor SimpleTensor::operator+=(const SimpleTensor& t1) {
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << "\n";
    if(_empty) {
        // here weak references and etc
        // _weak_ref_count = new size_t{0};
        _strong_ref_count = new size_t{1};
        _size = t1._size;
        _all_elements = t1._all_elements;
        _data = new float[t1._all_elements]{ 0 };
        _empty = false;
    }

    if( t1._size != _size )
        throw WrongDimensionsException(
            "Dimensions do not match for operation (+=). Trying to "
            + str_representation(_size)
            + " += "
            + str_representation(t1._size)
            + "\n"
        );

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        _data[i] += t1._data[i];

    return (*this);
}


SimpleTensor operator+(const SimpleTensor& t1, const SimpleTensor& t2) {
    if( t1._size != t2._size )
        throw WrongDimensionsException(
            "Dimensions do not match for operation (+). Trying to "
            + str_representation(t1._size)
            + " + "
            + str_representation(t2._size)
            + "\n"
        );

    // data of result tensor
    float* t3_data = new float[t1._all_elements];

    // adding tensors
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = t1._data[i] + t2._data[i];

    return SimpleTensor(t1._size, t3_data, false);
}


SimpleTensor operator*(float sc, SimpleTensor& t1) {
    // data of result tensor
    float* t3_data = new float[t1._all_elements];
    for(int i = 0; i < t1._all_elements; i++)
        t3_data[i] = sc * t1._data[i] ;

    return SimpleTensor(t1._size, t3_data, false);
}


SimpleTensor operator*(const SimpleTensor& t1, const SimpleTensor& t2) {
    std::vector<size_t> s1 = t1._size;
    std::vector<size_t> s2 = t2._size;
    // if t2 is vector then it should be made transformed [d, 1]
    
    // check for dim conditions
    size_t last_dim = s1[s1.size()-1];
    if( last_dim != s2[0]) // [a, b, ..., c, d] * [d, e] => [a, b, ..., c, e] 
        throw WrongDimensionsException(
            "Dimensions do not match for operation (*). Trying to "
            + str_representation(t1._size)
            + " * "
            + str_representation(t2._size)
            + "\n"
        );

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

    return SimpleTensor(s3, t3_data, false);
}


inline void SimpleTensor::evaluateCummulatives() {
    _cummulative_size = std::vector<size_t>(_size.size());
    for(int i = 0; i < _size.size(); i++) {
        size_t cummulation = 1;
        for(int j = i+1; j < _size.size(); j++)
            cummulation *= _size[j];
        _cummulative_size[i] = cummulation;
    }
}


inline void SimpleTensor::evaluateId() {
    std::stringstream ss;
    ss << _data;
    std::string id = ss.str();
    id += '#';

    for(auto el : _size)
        id += std::to_string(el);    
    _id = id;
}


inline std::vector<size_t> SimpleTensor::getSize() const {
    return _size;
}


void SimpleTensor::trim() {
    int len = _size.size();
    if( len > 2 && _size[len-1] == 1 && _size[len-2] == 1 ){
        _size.pop_back();
        _size.pop_back();
    }
}

void SimpleTensor::fill( SimpleTensor data_source) {
    if( _size != data_source._size )
        throw WrongSizeException(
            "Dimensions do not match for operation (*). Trying to "
            + str_representation(_size)
            + " * "
            + str_representation(data_source._size)
            + "\n"
        );

    for(int i = 0; i < data_source._all_elements; i++) {
        _data[i] = data_source._data[i];
    }
}


SimpleTensor& SimpleTensor::reshape(std::vector<size_t> new_size) {
    _size = new_size;
    evaluateCummulatives();
    return (*this);
} // needs a check


SimpleTensor SimpleTensor::slice(size_t start, size_t end) {  // needs to be more roboust
    SimpleTensor t1;
    std::vector< size_t> r_size(_size);
    r_size[0] = end - start;
    // std::cout << __PRETTY_FUNCTION__ << " " << (std::string) (*this) << " w_num: " << _weak_ref_count << "\n";

    // some kind of constructor would be better
    t1._size = r_size;
    t1._data = _data + start * _cummulative_size[0];
    t1._id = _id + "_s";
    t1._empty = false;
    t1._slice = true;
    t1._all_elements = r_size[0] * _cummulative_size[0];
    t1._strong_ref_count = _strong_ref_count;
    // t1._weak_ref_count = _weak_ref_count;
    // (*_weak_ref_count)++;
    t1.evaluateCummulatives();
    return t1;
}


SimpleTensor SimpleTensor::copy() {
    float* t_data = new float[this -> _all_elements];
    std::vector<size_t> t_size(this -> _size);

    for(int i = 0; i < this -> _all_elements; i++)
        t_data[i] = this -> _data[i];

    return SimpleTensor(t_size, t_data);
}


float SimpleTensor::at(std::vector<size_t> point) const {
    int flat_idx = 0;
    for(int i = 0; i < _size.size(); i++)
        flat_idx += point[i] * _cummulative_size[i];

    return *(_data + flat_idx);
}

void SimpleTensor::set(float val, std::vector<size_t> point) {
    int flat_idx = 0;
    for(int i = 0; i < _size.size(); i++)
        flat_idx += point[i] * _cummulative_size[i];

    *(_data + flat_idx) = val;    
}


SimpleTensor SimpleTensor::identity(size_t size) {
    SimpleTensor tens = SimpleTensor({size, size}, 0.0);
    for(int i = 0; i < size; i++)
        tens._data[i + i * size] = 1;
    return tens;
}


SimpleTensor SimpleTensor::rand(std::vector<size_t> size, std::pair<float, float> range) {
    SimpleTensor tens({size}, 0.0);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for(int i = 0; i < tens._all_elements; i++)
        tens._data[i] = distribution(SimpleTensor::rand_engine);

    return tens;
}


void print_rec(std::ostream& os, const SimpleTensor tensor, int depth) {
    // std::cout << __PRETTY_FUNCTION__ << str_representation(tensor.getSize()) << " - " << depth << "\n";
    std::vector<size_t> size( tensor.getSize() );

    if( size.size() == 1 ) {
        for(int k = 0; k < depth; k++)       
            os << " ";
        os << "[ ";
        for(unsigned int i = 0; i < size[0]; i++)
            // os << 0 << " ";
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
    os << "(size";
    print_size(os, tensor.getSize());
    os << ")";
    return os;
}