#ifndef __SIMPLE_TENSOR__
#define __SIMPLE_TENSOR__

#include <vector>

class SimpleTensor {

    friend class TensorOperations;
    friend SimpleTensor operator*(SimpleTensor& t1, SimpleTensor& t2);
    friend SimpleTensor operator*(float sc, SimpleTensor& t1);
    friend SimpleTensor operator+(SimpleTensor& t1, SimpleTensor& t2);
    friend class Model;
    
    public:
        SimpleTensor(); // no id. Just empty representation of a tensor. Not much can be done with it

        SimpleTensor(float value);

        SimpleTensor(std::vector<size_t> size, float init);

        SimpleTensor(std::vector<size_t> size, std::vector<float> data);

        SimpleTensor(std::vector<size_t> size, float* data, bool slice=false);

        SimpleTensor(SimpleTensor&& to_move);

        SimpleTensor& operator=(SimpleTensor && to_move);

        ~SimpleTensor();

        static SimpleTensor* identity(size_t size);

        static SimpleTensor* rand(std::vector<size_t> size, std::pair<float, float> range);

        SimpleTensor operator[] (size_t idx);

        const SimpleTensor operator[] (size_t idx) const;

        SimpleTensor operator+= (SimpleTensor t1);

        SimpleTensor reshape(std::vector<size_t> new_size) { _size = new_size; return (*this); } // needs a check

        void evaluateCummulatives();

        void evaluateId();

        void trim();

        SimpleTensor copy(const SimpleTensor to_copy) {
            return SimpleTensor(to_copy);
        }

        SimpleTensor slice(size_t start, size_t end) {  // needs to be more roboust
            std::vector< size_t> r_size(_size);
            r_size[0] = end - start;
            return SimpleTensor(r_size, _data + start * _cummulative_size[0], true);
        }

        // std::string getId() { return _id; }

        operator std::string() const { return _id; };

        std::vector<size_t> getSize() const;

        size_t nElements() const { return _all_elements; }

        float at(std::vector<size_t> point) const;

    protected:
        std::vector<size_t> _cummulative_size;
        std::vector<size_t> _size;
        bool _empty{false};

        std::string _id;
        size_t _all_elements {0};
        float* _data {nullptr};
        bool _slice;   

    private:
        // copy constructor shuld never be used in default.
        SimpleTensor(const SimpleTensor& to_copy);  

};

SimpleTensor operator*(float sc, SimpleTensor& t1);

SimpleTensor operator*(SimpleTensor& t1, SimpleTensor& t2);

SimpleTensor operator+(SimpleTensor& t1, SimpleTensor& t2);

void print_rec(std::ostream& os, const SimpleTensor& tensor, int depth=0);

void print_size(std::ostream& os, const std::vector<size_t> vec);

std::ostream& operator<< (std::ostream& os, const SimpleTensor& tensor);

#endif