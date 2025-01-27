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
#include "except.h"

void generateLossLandscape(Model& model, std::pair<float, float> range1, std::pair<float, float> range2);

int main() {

    Model model({
        new Layer(2, 1, false, "")
    });

    generateLossLandscape(model, {0, 4}, {-2, 2});


    Optimizer optim(model, 0.0005);

    size_t batch_size = 64;
    SimpleTensor sample;
    SimpleTensor y_real;
    Tensor y_pred, loss;

    SimpleTensor sam;
    SimpleTensor y_r;
    Tensor y_p, l;

    Dataloader dataloader("datasets/TwoClassProblem/at2po30.csv", batch_size, true);
    Dataloader dataloader_check("datasets/TwoClassProblem/at2po30.csv", batch_size, false);


    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream fout("learning.csv");
    int num = 0;
    float avg_loss = 0, sum_loss = 0;

    // starting point loss  
    sum_loss = 0;
    for(Batch batch_check : dataloader_check) {
        for(int i = 0; i < batch_check.x.getSize()[0]; i++) {
            y_r = batch_check.y[i].reshape({1, 1});
            sam = batch_check.x[i].reshape({2, 1});

            y_p = model(sam);
            l = TensorOperations::bceLoss(y_p, y_r);

            sum_loss += l.at({0, 0});
            l.getGraphContext() -> clearSequence();
        }
    }
    fout << model.getLayer("layer_0") -> getWeight().at({0, 0}) << ";" << model.getLayer("layer_0") -> getWeight().at({0, 1})<< ";" << sum_loss << "\n";


    for(int epoch = 0; epoch < 100; epoch++) {
        for(Batch batch : dataloader) {
            num = 0;
            avg_loss = 0;
            for(int i = 0; i < batch.x.getSize()[0]; i++) {
                y_real = batch.y[i].reshape({1, 1});
                sample = batch.x[i].reshape({2, 1});

                y_pred = model(sample);
                loss = TensorOperations::bceLoss(y_pred, y_real);

                // std::cout << "y_true: " << y_real.at({0, 0}) << " y_pred: [" << y_pred.at({0, 0}) << "] " << " loss : " << loss.at({0, 0}) << "\n"; 

                avg_loss += loss.at({0, 0});
                loss.getGraphContext() -> backwards();
                optim.step();
                loss.getGraphContext() -> clearSequence();

                num++;

                sum_loss = 0;
                for(Batch batch_check : dataloader_check) {
                    for(int j = 0; j < batch_check.x.getSize()[0]; j++) {
                        y_r = batch_check.y[j].reshape({1, 1});
                        sam = batch_check.x[j].reshape({2, 1});

                        y_p = model(sam);
                        l = TensorOperations::bceLoss(y_p, y_r);

                        sum_loss += l.at({0, 0});
                        l.getGraphContext() -> clearSequence();
                    }
                }
                fout << model.getLayer("layer_0") -> getWeight().at({0, 0}) << ";" << model.getLayer("layer_0") -> getWeight().at({0, 1})<< ";" << sum_loss << "\n";

            }
        }
        std::cout << "avg_loss: " << avg_loss / num << "\n";
    }

    std::cout << "final weights: " << model.getLayer("layer_0")->getWeight().at({0, 0}) << ", " << model.getLayer("layer_0")->getWeight().at({0, 1}) << "\n\n";
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time: " << duration.count() << " miliseconds\n";

}





//////////////////////////////////////// VISUALISATION OF REGRESSION

void generateLossLandscape(Model& model, std::pair<float, float> range1, std::pair<float, float> range2) {
    float step = 0.08;
    float w1, w2;

    Dataloader dataloader("datasets/TwoClassProblem/at2po30.csv", 1, false);

    float min_w1, min_w2, min_loss = 10e10;
    float max_loss = -1, avg_loss;
    int steps_all =  0;

    std::ofstream fout("loss_landscape.csv");
    for(float w1 = range1.first; w1 < range1.second; w1 += step)
        for(float w2 = range2.first; w2 < range2.second; w2 += step) {
            
            float loss_sum = 0;
            SimpleTensor sample;
            Tensor loss, y_real, y_pred;

            SimpleTensor weight_mod({1, 2}, {w1, w2});
            model.getLayer("layer_0") -> setWeight(weight_mod);

            for(Batch batch : dataloader) 
                for(int i = 0; i < batch.x.getSize()[0]; i++) {
                    y_real = batch.y[i].reshape({1, 1});
                    sample = batch.x[i].reshape({2, 1});

                    y_pred = model(sample);
                    loss = TensorOperations::bceLoss(y_pred, y_real);

                    loss_sum += loss.at({0, 0});
                    loss.getGraphContext() -> clearSequence();
                }    
            // std::cout << "w1: " << w1 << " w2: " << w2 << "  " << loss_sum << "\n";
            fout << w1 << ";" << w2 << ";" << loss_sum << "\n";

            if(loss_sum < min_loss) {
                min_w1 = w1;
                min_w2 = w2;
                min_loss = loss_sum;
            } else if (loss_sum > max_loss ) {
                max_loss = loss_sum;
            }
            avg_loss += loss_sum;
            steps_all++;

        }
    fout.clear();
    fout.close();
    std::cout << "max loss: " << max_loss << " avg loss: " << avg_loss /  steps_all << " |  minimum w1:" <<  min_w1 << " w2: " << min_w2 << " min_loss: " << min_loss << "\n";
}


// int main() {

//     Model model({
//         new Layer(2, 1, false, ""),
//     });
//     model["layer_0"] -> setWeight( SimpleTensor({2, 1}, {1, 1.8}) );

//     // generateLossLandscape(model, {-2, 2});
    
//     Optimizer optim(model, 0.07);

//     size_t batch_size = 10;
//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss;

//     SimpleTensor sam;
//     SimpleTensor y_r;
//     Tensor y_p, l;


//     Dataloader dataloader("datasets//RegressionProblem//at2po30.csv", batch_size, true);
//     Dataloader dataloader_check("datasets//RegressionProblem//at2po30.csv", batch_size, false);


//     auto start = std::chrono::high_resolution_clock::now();

//     std::ofstream fout("learning.csv");
//     int num = 0;
//     float avg_loss = 0, sum_loss = 0;

//     // starting point loss  
//     sum_loss = 0;
//     for(Batch batch_check : dataloader_check) {
//         for(int i = 0; i < batch_check.x.getSize()[0]; i++) {
//             y_r = batch_check.y[i].reshape({1, 1});
//             sam = batch_check.x[i].reshape({2, 1});

//             y_p = model(sam);
//             l = TensorOperations::mseLoss(y_p, y_r);

//             sum_loss += l.at({0, 0});
//             l.getGraphContext() -> clearSequence();
//         }
//     }
//     fout << model.getLayer("layer_0") -> getWeight().at({0, 0}) << ";" << model.getLayer("layer_0") -> getWeight().at({0, 1})<< ";" << sum_loss << "\n";


//     // learning
//     for(int epoch = 0; epoch < 1; epoch++) {
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
//         std::cout << epoch << ". avg_loss: " << avg_loss / num << "\n";
//     }
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
//     fout.clear();
//     fout.close();

//     std::cout << "return 0\n";
//     return 0;
// }





////////////////////////////////////////////////// Learning

// int main() {
//     Model model({
//         new Layer(2, 4, false, ""),
//         new Layer(4, 8, false, ""),
//         new Layer(8, 16, false, ""),
//         new Layer(16, 1, false, "")
//     });

//     Optimizer optim(model, 0.0001);

//     size_t batch_size = 10;
//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss;

//     Dataloader dataloader("datasets/RegressionProblem/at2po30.csv", batch_size, false);

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
//                 break;
//             }
//             break;
//         }
//         std::cout << "avg_loss: " << avg_loss / num << "\n";
//         break;
//     }
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
// }

// // 50 samples dataset , 50 epochs (2, 16, relu) (16, 3, relu) (3, 1, relu)
// // 1) 12.5s 13s 13.5s
// // 2) 0.9s

