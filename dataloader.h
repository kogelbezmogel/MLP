# ifndef DATALOADER
# define DATALOADER

#include <iostream>
#include <random>
#include <functional>
#include "simple_tensor.h"
#include "dataset.h"

struct Batch {
    Batch() { }
    Batch(SimpleTensor x, SimpleTensor y) {
        this->x = x;
        this->y = y;
    }

    SimpleTensor x;
    SimpleTensor y;
};


class Dataloader {

    private:
        Dataset _dataset;
        size_t _batch_size;
        std::vector< Batch > _batches;

        void generateBatches();

    public:

        Dataloader(Dataset dataset, int batch_size=32, bool shuffle=true);

        using Iterator = std::vector<Batch>::iterator;

        Iterator begin() { return _batches.begin(); }

        Iterator end() { return _batches.end(); }
};


# endif //DATALOADER 