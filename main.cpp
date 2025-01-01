#include <iostream>
#include <vector>
#include <chrono>

#include "tensor.h"
#include "optimizer.h"
#include "model.h"
#include "utils.h"
#include "dataloader.h"
#include "simple_tensor.h"



int main() {
    Model model({
        new Layer(2, 16, true, "relu"),
        new Layer(16, 3, true, "relu"),
        new Layer(3, 1, true, "relu")
    });

    Optimizer optim(model, 0.0001);

    size_t batch_size = 10;
    SimpleTensor sample;
    SimpleTensor y_real;
    Tensor y_pred, loss;

    Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", batch_size, false);

    auto start = std::chrono::high_resolution_clock::now();

    int num = 0;
    float avg_loss = 0;
    for(int epoch = 0; epoch < 50; epoch++) {
        for(Batch batch : dataloader) {
            num = 0;
            avg_loss = 0;
            for(int i = 0; i < batch.x.getSize()[0]; i++) {
                y_real = batch.y[i].reshape({1, 1});
                sample = batch.x[i].reshape({2, 1});

                y_pred = model(sample);
                loss = TensorOperations::mseLoss(y_pred, y_real);

                avg_loss += loss.at({0, 0});
                loss.getGraphContext() -> backwards();
                optim.step();
                loss.getGraphContext() -> clearSequence();

                num++;
                // break;
            }
            // break;
        }
        std::cout << "avg_loss: " << avg_loss / num << "\n";
        // break;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time: " << duration.count() << " miliseconds\n";


    // SimpleTensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
    // SimpleTensor t2({2, 3}, {2, 4, 6, 8, 10, 12});

    // std::cout << t1 << "\n";
    // std::cout << t2 << "\n";
    // std::cout << t1 * t2 << "\n";

    std::cout << "return 0\n";
    return 0;
}


// void full_test() { 
//     Model model({
//         new Layer(2, 16, true, "relu"),
//         new Layer(16, 3, true, "relu"),
//         new Layer(3, 1, true, "relu")
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
//     for(int epoch = 0; epoch < 50; epoch++) {
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
//                 // break;
//             }
//             // break;
//         }
//         std::cout << "avg_loss: " << avg_loss / num << "\n";
//         // break;
//     }
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
// }


// 50 samples dataset , 50 epochs (2, 16, relu) (16, 3, relu) (3, 1, relu)
// 1) 12.5s 13s 13.5s
// 2) 0.9s

