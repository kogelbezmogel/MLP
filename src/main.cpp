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

using lossFun = Tensor(*) (Tensor t1, SimpleTensor t2);


int main() {

    // definig the model
    Model model({
        new Layer(4, 16, true, "relu"),
        new Layer(16, 16, true, "relu"),
        new Layer(16, 3, false, "")
    });

    size_t batch_size = 16;

    // creating optimizer 
    BGD optim(model, batch_size, 0.01);
    // loading dataset
    Dataset dataset("datasets//Iris//iris.csv", ',', true, 0.7, true);
    // train dataloader
    Dataloader dataloader_train(dataset.get_train_set(), batch_size, true);
    // test dataloader
    Dataloader dataloader_test(dataset.get_test_set(), batch_size, false);

    // file to save training history
    std::ofstream history_file("iris_200e.csv");
    history_file << "epoch,train_loss,test_loss,train_acc,test_acc\n"; 

    auto start = std::chrono::high_resolution_clock::now();

    // mina training loop
    for(int epoch = 0; epoch < 200; epoch++) {
        // resetting metrics
        int train_num = 0;
        float train_loss = 0;
        float train_accuracy = 0;
            
        // updating weights
        for(Batch batch : dataloader_train) {
            for(int i = 0; i < batch.x.getSize()[0]; i++) {
                SimpleTensor y_real = batch.y[i].reshape({1, 1});
                SimpleTensor sample = batch.x[i].reshape({4, 1});

                Tensor y_pred = model(sample);
                Tensor loss = TensorOperations::cceLoss(y_pred, y_real);

                train_loss += loss.at({0, 0});
                loss.getGraphContext() -> backwards();
                optim.step();
                loss.getGraphContext() -> clearSequence();

                train_num++;
                if(y_pred.maxInd() == y_real.at({0, 0}))
                    train_accuracy++;
            }
        }
        optim.close();

        // gathering metrics about test data
        float test_accuracy = 0, test_loss = 0;
        size_t test_num = 0;
        SimpleTensor y_r, sam;
        Tensor l, y_p;
        for(Batch batch_check : dataloader_test) {
            for(int i = 0; i < batch_check.x.getSize()[0]; i++) {
                y_r = batch_check.y[i].reshape({1, 1});
                sam = batch_check.x[i].reshape({4, 1});

                y_p = model(sam);

                if(y_p.maxInd() == y_r.at({0, 0}))
                    test_accuracy++;

                l = TensorOperations::cceLoss(y_p, y_r);
                test_loss += l.at({0, 0});

                l.getGraphContext() -> clearSequence();
                test_num++;
            }
        }

        // printing metrics to console
        std::cout << epoch
                  << " | tr_loss: " << train_loss / train_num
                  << " te_loss: " << test_loss / test_num
                  << " tr_acc: " << train_accuracy / train_num
                  << " te_acc: " << test_accuracy / test_num << "\n";
        
        // writting metrics to file
        history_file << epoch
                     << "," << train_loss / train_num
                     << "," << test_loss / test_num
                     << "," << train_accuracy / train_num
                     << "," << test_accuracy / test_num << "\n";

    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time: " << duration.count() << " miliseconds\n";
}
