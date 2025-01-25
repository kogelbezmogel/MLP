# ifndef DATALOADER
# define DATALOADER

#include <iostream>
#include <random>
#include <functional>
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
        char _sep;
        std::string _file_path;
        bool _header;
        SimpleTensor _data_x;
        SimpleTensor _data_y;
        size_t _batch_size;
        std::vector< Batch > _batches;


        static std::mt19937 _rand_engine;

        void loadDataFromCSV();

        void generateBatches();

        void head(int rows = 5);

        void tail(int rows = 5);

        int countColumns();

    public:

        using Iterator = std::vector<Batch>::iterator;

        Dataloader(std::string file_path, int batch_size=32, bool shuffle=true, char sep=',', bool header=false);

        Iterator begin() { return _batches.begin(); }

        Iterator end() { return _batches.end(); }

        void shuffle();

    template<typename T>
    static std::vector<T> parseLine(std::string line, char sep, std::function<T(std::string)> cast_fun) {
        std::vector<T> parsed_line;

        std::string word = "";
        for(std::string::iterator it = line.begin(); it != line.end(); it++){
                if((*it) == sep) {
                    parsed_line.push_back( cast_fun(word) );
                    word.clear();
                }
                else if(it == line.end()-1) {
                    word.append( std::string{*it} );
                    parsed_line.push_back( cast_fun(word) );
                    word.clear();
                } else {
                    word.append( std::string{*it} );
                }
            }
            return parsed_line;
    }
};


# endif //DATALOADER 