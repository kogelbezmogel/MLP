#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "tensor.h"
#include "optimizer.h"
#include "model.h"
#include "utils.h"
#include "dataloader.h"
#include "simple_tensor.h"


int main() {
    SimpleTensor t1({2, 2}, {1, 2, 3, 4});
    SimpleTensor t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    std::cout << t1 * t2;
}


// void generateLossLandscape(Model& model, std::pair<float, float> range) {
//     float step = 0.1;
//     float w1, w2;

//     Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", 1, false);

//     float min_w1, min_w2, min_loss = 10e10;

//     std::ofstream fout("loss_landscape.csv");
//     for(float w1 = range.first; w1 < range.second; w1 += step)
//         for(float w2 = range.first; w2 < range.second; w2 += step) {
            
//             float loss_sum = 0;
//             SimpleTensor sample;
//             Tensor loss, y_real, y_pred;

//             SimpleTensor weight_mod({1, 2}, {w1, w2});
//             model.getLayer("layer_0") -> setWeight(weight_mod);

//             for(Batch batch : dataloader) 
//                 for(int i = 0; i < batch.x.getSize()[0]; i++) {
//                     y_real = batch.y[i].reshape({1, 1});
//                     sample = batch.x[i].reshape({2, 1});

//                     y_pred = model(sample);
//                     loss = TensorOperations::mseLoss(y_pred, y_real);

//                     loss_sum += loss.at({0, 0});
//                     loss.getGraphContext() -> clearSequence();
//                 }    
//             // std::cout << "w1: " << w1 << " w2: " << w2 << "  " << loss_sum << "\n";
//             fout << w1 << ";" << w2 << ";" << loss_sum << "\n";

//             if(loss_sum < min_loss) {
//                 min_w1 = w1;
//                 min_w2 = w2;
//                 min_loss = loss_sum;
//             }
//         }
//     fout.clear();
//     fout.close();
//     std::cout << "minimum w1:" <<  min_w1 << " w2: " << min_w2 << " min_loss: " << min_loss << "\n";
// }



// int main() {


//     Model model({
//         new Layer(2, 1, false, "")
//     });

//     generateLossLandscape(model, {-2, 2});

//     Optimizer optim(model, 0.03);

//     size_t batch_size = 10;
//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss;

//     SimpleTensor sam;
//     SimpleTensor y_r;
//     Tensor y_p, l;



//     Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", batch_size, false);
//     Dataloader dataloader_check("datasets\\RegressionProblem\\at2po30.csv", batch_size, false);


//     auto start = std::chrono::high_resolution_clock::now();

//     std::ofstream fout("learning.csv");
//     int num = 0;
//     float avg_loss = 0, sum_loss = 0;
//     for(int epoch = 0; epoch < 2; epoch++) {
//         for(Batch batch : dataloader) {
//             num = 0;
//             avg_loss = 0;
//             for(int i = 0; i < batch.x.getSize()[0]; i++) {
//                 y_real = batch.y[i].reshape({1, 1});
//                 sample = batch.x[i].reshape({2, 1});

//                 y_pred = model(sample);
//                 loss = TensorOperations::mseLoss(y_pred, y_real);

//                 avg_loss += loss.at({0, 0});
//                 loss.getGraphContext() -> backwards();
//                 optim.step();
//                 loss.getGraphContext() -> clearSequence();
//                 num++;


//                 sum_loss = 0;
//                 for(Batch batch_check : dataloader_check) {
//                     for(int i = 0; i < batch_check.x.getSize()[0]; i++) {
//                         y_r = batch_check.y[i].reshape({1, 1});
//                         sam = batch_check.x[i].reshape({2, 1});

//                         y_p = model(sam);
//                         l = TensorOperations::mseLoss(y_p, y_r);

//                         sum_loss += l.at({0, 0});
//                         l.getGraphContext() -> clearSequence();
//                     }
//                 }
//                 fout << model.getLayer("layer_0") -> getWeight().at({0, 0}) << ";" << model.getLayer("layer_0") -> getWeight().at({0, 1})<< ";" << sum_loss << "\n";

//             }
//         }
//         std::cout << "avg_loss: " << avg_loss / num << "\n";
//     }
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
//     fout.clear();
//     fout.close();

//     std::cout << "return 0\n";
//     return 0;
// }

// void test() {
//     Model model({
//         new Layer(2, 1, "")
//     });

//     Optimizer optim(model, 0.0001);

//     size_t batch_size = 10;
//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss;

//     Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", batch_size, false);

//     auto start = std::chrono::high_resolution_clock::now();

//     int num = 0;
//     float avg_loss = 0;
//     for(int epoch = 0; epoch < 3000; epoch++) {
//         for(Batch batch : dataloader) {
//             num = 0;
//             avg_loss = 0;
//             for(int i = 0; i < batch.x.getSize()[0]; i++) {
//                 y_real = batch.y[i].reshape({1, 1});
//                 sample = batch.x[i].reshape({2, 1});

//                 y_pred = model(sample);
//                 loss = TensorOperations::mseLoss(y_pred, y_real);

//                 avg_loss += loss.at({0, 0});
//                 loss.getGraphContext() -> backwards();
//                 optim.step();
//                 loss.getGraphContext() -> clearSequence();

//                 num++;
//             }
//         }
//         std::cout << "avg_loss: " << avg_loss / num << "\n";
//     }
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
// }

// // 50 samples dataset , 50 epochs (2, 16, relu) (16, 3, relu) (3, 1, relu)
// // 1) 12.5s 13s 13.5s
// // 2) 0.9s

