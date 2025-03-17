#include "dataloader.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>


Dataloader::Dataloader(Dataset dataset, int batch_size, bool shuffle) :
_batch_size(batch_size),
_dataset(dataset) {
    generateBatches();
}


void Dataloader::generateBatches() {
    size_t data_length = _dataset._data_x.getSize()[0];
    int batches_num = std::ceil( (float) data_length / _batch_size );
    // std::cout << data_length << " " << _batch_size << " " << batches_num;
    
    for(int batch_id = 0; batch_id < batches_num; batch_id++) {
        size_t start, end;
        start = batch_id*_batch_size;
        end = std::min( (batch_id+1)*_batch_size, data_length );
    
        _batches.push_back( Batch(_dataset._data_x.slice(start, end), _dataset._data_y.slice(start, end)) );
    }
}