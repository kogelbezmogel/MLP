#ifndef __DATASET__
#define __DATASET__

#include <functional>
#include <cmath>
#include "except.h"
#include "simple_tensor.h"

class Dataset {

    friend class Dataloader;

    private:
        SimpleTensor _data_x;
        SimpleTensor _data_y;
        size_t _samples_num;
        char _sep;
        std::string _file_path;
        float _split_train_test;
        bool _header;
        bool _shuffle;
        static std::mt19937 _rand_engine;

        
    public:
        Dataset(std::string file_path, char sep=',', bool header=false, float split_train_test=1.0, bool shuffle=true);
        Dataset(SimpleTensor data_x, SimpleTensor data_y);

        Dataset get_train_set();

        Dataset get_test_set();

        void loadDataFromCSV();

        void head(int rows = 5);

        void tail(int rows = 5);

        void shuffle();

        int countColumns();

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


#endif