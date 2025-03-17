#include <iostream>
#include <fstream>
#include "dataset.h"

std::mt19937 Dataset::_rand_engine = std::mt19937{ (long unsigned int) time(NULL) % 101 };

Dataset::Dataset(std::string file_path, char sep, bool header, float split_train_test, bool shuffle) : 
_header(header),
_sep(sep),
_file_path(file_path),
_split_train_test(split_train_test),
_shuffle(shuffle) {
    loadDataFromCSV();
    if(shuffle)
        this -> shuffle();
}


Dataset::Dataset(SimpleTensor data_x, SimpleTensor data_y) : 
_data_x(data_x),
_data_y(data_y),
_header(false),
_sep('-'),
_file_path(""),
_shuffle(false) {
}


void Dataset::loadDataFromCSV() {
    size_t columns_count = countColumns();
    std::ifstream fin(_file_path, std::ios_base::in);
    std::vector<std::vector<float>> data;

    // reading file line by line
    std::string line;
    std::vector<float> parsed_line;

    if(_header) {
        std::getline(fin, line);
        // std::cout << "headers: " << line << "\n";
    }

    while(std::getline(fin, line)) {
        parsed_line = parseLine<float>(line, _sep, [] (std::string val) {return std::stof(val); });
        data.push_back(parsed_line);
    }
    fin.clear();
    fin.close();

    _samples_num = data.size();

    std::vector<size_t> data_size{data.size(), data[0].size()};
    std::vector<float> data_x;
    std::vector<float> data_y;
    for(std::vector<float> vec : data) {
        data_y.push_back(vec[0]);
        for(int i = 1; i < columns_count; i++)
            data_x.push_back(vec[i]);
        // std::cout << vec[0] << " " << vec[1] << " " << vec[2] << "\n";
    }

    _data_x = SimpleTensor({data_size[0], columns_count-1}, data_x);
    _data_y = SimpleTensor({data_size[0], 1}, data_y);
}


void Dataset::head(int rows) {
    for( int i = 0; i < rows; i++) {
        std::cout << _data_x[i] << " | " << _data_y[i] << "\n";
    }
}


void Dataset::tail(int rows) {
    for( int i = 0; i < rows; i++) {
            std::cout << _data_x[ _data_x.getSize()[0]-rows-1 + i ] << " | " << _data_y[_data_x.getSize()[0]-rows-1 + i];
    }
}


int Dataset::countColumns() {
    std::ifstream file(_file_path, std::ios_base::in);
    std::string line;
    int columns_count;

    std::getline(file, line);
    std::getline(file, line); // to ommit header column
    std::vector<std::string> vals = parseLine<std::string>( line, _sep, [] (std::string val) {return val;});
    columns_count = vals.size();

    file.clear();
    file.close();

    return columns_count;
}


Dataset Dataset::get_train_set() {
    size_t train_set_length = std::round(_split_train_test * _samples_num);
    size_t start = 0;
    size_t end = start + train_set_length;

    std::cout << start << " - " << end << " " << _split_train_test << "\n";

    return Dataset(_data_x.slice(start, end), _data_y.slice(start, end));
}


Dataset Dataset::get_test_set() {
    size_t test_set_length = std::round((1-_split_train_test) * _samples_num);
    size_t start = std::round(_split_train_test * _samples_num);
    size_t end = start + test_set_length;

    std::cout << start << " - " << end << "\n";

    if(test_set_length <= 0)
        throw  NoTestDataInDataset("No test samples in dataset with split ratio " + std::to_string(_split_train_test));
    return Dataset(_data_x.slice(start, end), _data_y.slice(start, end));
}


void Dataset::shuffle() {
    size_t length = _data_x.getSize()[0];
    
    std::vector<size_t> indexes(length, 0);
    for(int i = 0; i < length; i++) {
        indexes[i] = i;
    }
    std::shuffle(indexes.begin(), indexes.end(), _rand_engine);

    SimpleTensor temp;
    for(int idx_1 = 0, idx_2; idx_1 < length; idx_1++) {
        idx_2 = indexes[idx_1];

        temp = _data_x[idx_1].copy();
        _data_x[idx_1].fill( _data_x[idx_2] );
        _data_x[idx_2].fill( temp );

        temp = _data_y[idx_1].copy();
        _data_y[idx_1].fill(_data_y[idx_2]);
        _data_y[idx_2].fill( temp );
    }
}