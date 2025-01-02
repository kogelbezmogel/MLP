#include "dataloader.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>


std::vector<float> parseLine(std::string line, char sep=',') {
    std::vector<float> parsed_line;

    std::string word = "";
    for(std::string::iterator it = line.begin(); it != line.end(); it++){
        if((*it) == sep) {
            parsed_line.push_back( std::stof(word) );
            word.clear();
        }
        else if(it == line.end()-1) {
            word.append( std::string{*it} );
            parsed_line.push_back( std::stof(word) );
            word.clear();
        } else {
            word.append( std::string{*it} );
        }
    }
    return parsed_line;
}

void Dataloader::loadDataFromCSV() {
    std::ifstream fin(_file_path, std::ios_base::in);
    std::vector<std::vector<float>> data;

    // reading file line by line
    std::string line;
    std::vector<float> parsed_line;
    while(std::getline(fin, line)) {
        parsed_line = parseLine(line);
        data.push_back(parsed_line);
    }
    fin.clear();
    fin.close();

    std::vector<size_t> data_size{data.size(), data[0].size()};
    std::vector<float> data_x;
    std::vector<float> data_y;
    for(std::vector<float> vec : data) {    // this need to be changed
        data_y.push_back(vec[0]);
        data_x.push_back(vec[1]);
        data_x.push_back(vec[2]);
        // std::cout << vec[0] << " " << vec[1] << " " << vec[2] << "\n";
    }

    _data_x = SimpleTensor({data_size[0], 2},  data_x);
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


Dataloader::Dataloader(std::string file_path, int batch_size, bool shuffle) : _batch_size(batch_size) {
    this -> _file_path = file_path;

    // std::cout << _file_path << " initialised" << std::endl; 
    loadDataFromCSV();

    if(shuffle)
        this -> shuffle();

    generateBatches();

}


void Dataloader::shuffle() {
    // size_t length = _data_x.getSize()[0];

    // for()
}