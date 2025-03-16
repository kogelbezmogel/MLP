#include "dataloader.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>


std::mt19937 Dataloader::_rand_engine = std::mt19937{ (long unsigned int) time(NULL) % 101 };


Dataloader::Dataloader(std::string file_path, int batch_size, bool shuffle, char sep, bool header) : _batch_size(batch_size), _sep(sep) {
    _header = header;
    this -> _file_path = file_path;

    std::cout << _file_path << " initialised" << std::endl; 
    loadDataFromCSV();
    if(shuffle)
        this -> shuffle();
 
    generateBatches();
}

void Dataloader::loadDataFromCSV() {
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


void Dataloader::generateBatches() {
    size_t data_length = _data_x.getSize()[0];
    int batches_num = std::ceil( (float) data_length / _batch_size );
    // std::cout << data_length << " " << _batch_size << " " << batches_num;
    
    for(int batch_id = 0; batch_id < batches_num; batch_id++) {
        size_t start, end;
        start = batch_id*_batch_size;
        end = std::min( (batch_id+1)*_batch_size, data_length );
    
        _batches.push_back( Batch(_data_x.slice(start, end), _data_y.slice(start, end)) );
    }
}


void Dataloader::shuffle() {
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


void Dataloader::head(int rows) {
    for( int i = 0; i < rows; i++) {
        std::cout << _data_x[i] << " | " << _data_y[i] << "\n";
    }
}


void Dataloader::tail(int rows) {
    for( int i = 0; i < rows; i++) {
            std::cout << _data_x[ _data_x.getSize()[0]-rows-1 + i ] << " | " << _data_y[ _data_x.getSize()[0]-rows-1 + i ];
    }
}


int Dataloader::countColumns() {
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