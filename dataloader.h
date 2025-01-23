# ifndef DATALOADER
# define DATALOADER

#include <iostream>
#include <random>
#include "simple_tensor.h"


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
        std::string _file_path;
        SimpleTensor _data_x;
        SimpleTensor _data_y;
        size_t _batch_size;
        std::vector< Batch > _batches;

        static std::mt19937 _rand_engine;

        void loadDataFromCSV();
        void generateBatches();

        void head(int rows = 5);
        void tail(int rows = 5);


    public:

        using Iterator = std::vector<Batch>::iterator;

        Dataloader(std::string file_path, int batch_size=32, bool shuffle=true);

        Iterator begin() { return _batches.begin(); }

        Iterator end() { return _batches.end(); }

        void shuffle();

};


# endif //DATALOADER 