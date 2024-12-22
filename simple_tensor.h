#ifndef __SIMPLE_TENSOR__
#define __SIMPLE_TENSOR__

#include <vector>

class SimpleTensor {

    friend class TensorOperations;
    friend SimpleTensor operator*(const SimpleTensor& t1, const SimpleTensor& t2);
    // friend SimpleTensor operator*(float sc, SimpleTensor& t1);
    // friend SimpleTensor operator+(SimpleTensor& t1, SimpleTensor& t2);
    friend class Model;
    
    public:
        SimpleTensor(std::vector<size_t> size, float init);

        SimpleTensor(const SimpleTensor& to_copy);  

        static SimpleTensor rand(std::vector<size_t> size, std::pair<float, float> range);
        
        ~SimpleTensor();

        SimpleTensor(std::vector<size_t> size, std::vector<float> data);

        SimpleTensor& operator=(const SimpleTensor& to_copy);

        SimpleTensor(); // no id. Just empty representation of a tensor. Not much can be done with it

        // SimpleTensor(float value);


        SimpleTensor(std::vector<size_t> size, float* data, bool slice=false);

        // SimpleTensor(SimpleTensor&& to_move);

        // SimpleTensor& operator=(SimpleTensor && to_move);


        // static SimpleTensor identity(size_t size);

        // SimpleTensor operator[] (size_t idx);

        // const SimpleTensor operator[] (size_t idx) const;

        SimpleTensor operator+= (const SimpleTensor& t1);

        // SimpleTensor reshape(std::vector<size_t> new_size) { _size = new_size; return (*this); } // needs a check

        void evaluateCummulatives();

        void evaluateId();

        void trim();

        // makes deep copy 
        // SimpleTensor copy(const SimpleTensor to_copy);

        // SimpleTensor slice(size_t start, size_t end) {  // needs to be more roboust
        //     SimpleTensor t1;
        //     std::vector< size_t> r_size(_size);
        //     r_size[0] = end - start;

        //     // some kind of constructor would be better
        //     t1._size = r_size;
        //     t1._data = _data + start * _cummulative_size[0];
        //     t1._id = _id + "_s";
        //     t1._empty = false;
        //     t1._slice = true;
        //     t1._all_elements = r_size[0] * _cummulative_size[0];
        //     t1._strong_ref_count = _strong_ref_count;
        //     t1._weak_ref_count = _weak_ref_count;
        //     (*_weak_ref_count)++;
        //     t1.evaluateCummulatives();
        //     return t1;
        // }

        operator std::string() const { return _id; };

        std::vector<size_t> getSize() const;

        size_t nElements() const { return _all_elements; }

        size_t getWeakCount() { return *_weak_ref_count; }

        size_t getStrongCount() { return *_strong_ref_count; }

        float* getDataPtr() { return _data; }

        float at(std::vector<size_t> point) const;

    protected:
        std::vector<size_t> _cummulative_size;
        std::vector<size_t> _size;
        bool _empty{false};

        std::string _id;
        size_t _all_elements {0};
        float* _data {nullptr};
        bool _slice;   

        // shared_pointer mechanism
        size_t* _strong_ref_count;
        size_t* _weak_ref_count;

        void __clean_up__();
};

// void print_rec(std::ostream& os, const SimpleTensor& tensor, int depth=0);

// void print_size(std::ostream& os, const std::vector<size_t> vec);

// std::ostream& operator<< (std::ostream& os, const SimpleTensor& tensor);

#endif