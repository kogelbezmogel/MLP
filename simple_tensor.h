#ifndef __SIMPLE_TENSOR__
#define __SIMPLE_TENSOR__

#include <vector>
#include <random>

class SimpleTensor {

    friend class TensorOperations;
    friend SimpleTensor operator*(const SimpleTensor& t1, const SimpleTensor& t2);
    friend SimpleTensor operator*(float sc, SimpleTensor& t1);
    friend SimpleTensor operator+(const SimpleTensor& t1, const SimpleTensor& t2);
    friend class Model;
    
    public:
        SimpleTensor(); // no id. Just empty representation of a tensor. Not much can be done with it

        SimpleTensor(std::vector<size_t> size, float init);

        SimpleTensor(const SimpleTensor& to_copy);  

        SimpleTensor(std::vector<size_t> size, std::vector<float> data);

        SimpleTensor(std::vector<size_t> size, float* data, bool slice=false);

        SimpleTensor(SimpleTensor&& to_move);

        ~SimpleTensor();

        SimpleTensor& operator=(const SimpleTensor& to_copy);

        SimpleTensor& operator=(SimpleTensor && to_move);

        SimpleTensor operator[] (size_t idx);

        const SimpleTensor operator[] (size_t idx) const;

        SimpleTensor operator+= (const SimpleTensor& t1);

        SimpleTensor& reshape(std::vector<size_t> new_size);

        void evaluateCummulatives();

        void evaluateId();

        void trim();

        // makes deep copy 
        SimpleTensor copy(const SimpleTensor to_copy);

        SimpleTensor slice(size_t start, size_t end);

        operator std::string() const { return _id; };

        std::vector<size_t> getSize() const;

        size_t nElements() const { return _all_elements; }

        float at(std::vector<size_t> point) const;

        void set(float val, std::vector<size_t> point);

        static SimpleTensor rand(std::vector<size_t> size, std::pair<float, float> range);

        static SimpleTensor identity(size_t size);


    protected:
        static std::mt19937 rand_engine;

        std::vector<size_t> _cummulative_size;
        std::vector<size_t> _size;
        bool _empty{false};

        std::string _id;
        size_t _all_elements {0};
        float* _data {nullptr};
        bool _slice;   

        // shared_pointer mechanism
        size_t* _strong_ref_count;
        // size_t* _weak_ref_count;

        void __clean_up__();
};

void print_rec(std::ostream& os, const SimpleTensor tensor, int depth=0);

void print_size(std::ostream& os, const std::vector<size_t> vec);

std::ostream& operator<< (std::ostream& os, const SimpleTensor& tensor);

#endif