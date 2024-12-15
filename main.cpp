#include <iostream>
#include <vector>
#include <chrono>

#include "tensor.h"
// #include "optimizer.h"
// #include "model.h"
#include "utils.h"
// #include "dataloader.h"

// SimpleTensor mul2(SimpleTensor& st1, SimpleTensor& st2) {
//     SimpleTensor st3;
//     st3 = st1 * st2;
//     return st3;
// }

// SimpleTensor mul1() {
//     SimpleTensor st1({2, 3}, {1, 2, 3, 4, 5, 6});
//     SimpleTensor st2({3, 2}, {1, 2, 3, 4, 5, 6});
//     SimpleTensor st3;
//     st3 = mul2(st1, st2);
//     return st3;
// }


int main() {
    // time_test_model();

    // mul();

    // SimpleTensor st1({2, 3}, {1, 2, 3, 4, 5, 6});
    // SimpleTensor st2({3, 2}, {1, 2, 3, 4, 5, 6});
    // // SimpleTensor st3 = st1 * st2;
    // SimpleTensor st3 = mul1();

    // std::cout << st3;

    // SimpleTensor st4 = st2.slice(0, 2);
    // SimpleTensor st5 = st4 * st3;

    // std::cout << st5 << "\n";

    Graph* graph = new Graph();
    Tensor t1({1, 3}, {1, 2, 3}, true, graph);
    Tensor t2({3, 1}, {1, 2, 3}, true, graph);

    Tensor t3 = TensorOperations::mul(t1, t2);
    // std::cout << t3;

    SimpleTensor real({1, 1}, {0.5});
    SimpleTensor loss = TensorOperations::mseLoss(t3, real);

    graph->saveGraphToFile("graph.dot");
    graph->backwards();

    std::cout << "return 0\n";
    return 0;
}


// 50 samples dataset , 50 epochs (2, 16, relu) (16, 3, relu) (3, 1, relu) -> 12.5s 13s 13.5s



// void time_test_model() {
//     Model model({
//         new Layer(2, 16, true, "relu"),
//         new Layer(16, 3, true, "relu"),
//         new Layer(3, 1, false, ""),
//     });

//     Optimizer optim(model, 0.001);

//     size_t batch_size = 10;
//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss_value;

//     auto start = std::chrono::high_resolution_clock::now();

//     int num = 0;
//     for(int epoch = 0; epoch < 50; epoch++) {
//         Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", batch_size, false);
//         for(Batch batch : dataloader) {
//             for(int i = 0; i < batch.x.getSize()[0]; i++) {
//                 y_real = batch.y[i].reshape({1, 1});
//                 sample = batch.x[i].reshape({2, 1});

//                 y_pred = model(sample);

//                 loss_value = TensorOperations::mseLoss(y_pred, y_real);
//                 loss_value.getGraphContext()->backwards();

//                 optim.step();
//                 // std::cout << num << " loss: " << std::sqrt(loss_value.at({0, 0})) << "\n";
//                 num++;
//             }
//         }
//     }

//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << "time: " << duration.count() << " miliseconds\n";
//     std::cout << "ret0";
// }


// int dataloader_test() {
//     Dataloader dataloader("datasets\\RegressionProblem\\at2po30.csv", 5, false);
//     for(int epoch = 0; epoch < 2; epoch++)
//         for(Batch batch : dataloader) {
//             std::cout << batch.y << "\n\n";
//         }

//     std::cout << "ret0";
// }


// void learning_test() {
//         size_t num_samples = 10;

//     // 10 samples each 1 value
//     SimpleTensor real({num_samples, 1}, {0.5});

//     // 10 training samples each has two attributes from range [-1, 2]
//     SimpleTensor training_samples( SimpleTensor::rand({num_samples, 3}, {-1, 2}) );

//     Model model({
//         new Layer(2, 3, false, "relu"),
//         new Layer(3, 3, false, "relu"),
//         new Layer(3, 1, true, ""),
//     });

//     Optimizer optim(model);

//     SimpleTensor sample;
//     SimpleTensor y_real;
//     Tensor y_pred, loss_value;

//     for(int i = 0; i < num_samples; i++) {
//         std::cout << "\n\n";
//         y_real = real[i].reshape({1, 1});
//         sample = training_samples[i].reshape({3, 1});

//         // model.getGraph() -> saveGraphToFile( "graph_sample" + std::to_string(i) + "_before.dot" );
//         y_pred = model(sample);
//         loss_value = TensorOperations::mseLoss(y_pred, y_real);
//         // loss_value.getGraphContext() -> saveGraphToFile( "graph_sample" + std::to_string(i) + "_after.dot" );
//         loss_value.getGraphContext()->backwards();

//         optim.step();
//     }
// }


// void gradient_tests() {
//     Tensor real({1, 1}, {0.5});

//     Model model({
//         new Layer(2, 3, true, "relu"),
//         new Layer(3, 3, true, "relu"),
//         new Layer(3, 1, true, ""),
//     });

//     TensorOperations::mseLoss( model( SimpleTensor::rand({2, 1}, {0, 1}) ), real );
//     model.getGraph() -> saveGraphToFile("graph3.dot");

//     std::cout << "\nReport\n\n";
//     std::map<std::string, float> report = model.gradient_report();
//     for(std::pair<std::string, float> layer_res : report)
//         std::cout << "layer: " << layer_res.first << " avg_err=" << layer_res.second << "\n";
// }



// void full_model_test() {
//         Tensor real({1, 1}, {0.5});

//     Model model({
//         new Layer(2, 3, true, "relu"),
//         new Layer(3, 3, true, ""),
//         new Layer(3, 1, true)
//     });

//     std::cout << "\n\n# Model created\n\n";

//     Tensor input = SimpleTensor::rand({2, 1}, {0, 1});
//     Tensor output = TensorOperations::mseLoss( model(input), real );

//     std::cout << "\n\n# Output given\n\n";
    
//     output.getGraphContext() -> saveGraphToFile("graph3.dot");
//     output.getGraphContext() -> backwards();

//     std::cout << "end";
// }


// void tensor_with_graph_test() {
    
//     Graph* g = new Graph();

//     Tensor x1 = Tensor(
//         {3, 1},
//         {1, 2, 3},
//         true,
//         g
//     ); 

//     Tensor w1 = Tensor(
//         {3, 3},
//         {1, 2, 3, 4, 5, 6, 7, 8, 9},
//         true,
//         g
//     );

//     Tensor b1 = Tensor(
//         {3, 1},
//         {5, 6, 7},
//         true,
//         g
//     );

//     Tensor w2 = Tensor(
//         {2, 3},
//         {1, 2, 3, 4, 5, 6},
//         true,
//         g
//     );

//     Tensor b2 = Tensor(
//         {2, 1},
//         {5, 6},
//         true,
//         g
//     );

//     Tensor x2 = TensorOperations::relu( TensorOperations::add(TensorOperations::mul(w1, x1), b1) );
//     Tensor x3 = TensorOperations::relu( TensorOperations::add(TensorOperations::mul(w2, x2), b2) );

//     g -> saveGraphToFile("graph2.dot");
//     g -> backwards();

//     std::cout << "end";
// }

// void tensor_test() {
//     Tensor t1(
//         {1, 2, 3, 4, 5, 6, 2, 2, 2, 2, 2, 2},
//         {2, 2, 3}
//     );
//     // std::cout << t1 << "\n";
    
//     Tensor  t2(
//         {1, 2, 3, 4, 5, 6, 7, 8, 9},
//         {3, 3}
//     );

//     std::cout << t2 << std::endl;
//     Tensor t3 = t1 * t2;
//     std::cout << t3 << std::endl;
    
//     t1 = Tensor(
//         {1, 2, 3, 4, 5, 6},
//         {2, 3}
//     );

//     t2 = Tensor(
//         {1, 2, 3, 4, 5, 6},
//         {2, 3}
//     );
// }



// void graph_test() {
//     Graph g;
//     Node* n0 = new Node(3);
//     Node* n1 = new Node(5);
//     std::vector<Node*> args({n0, n1});

//     Node* n2 = g["mul"](args);
//     args = std::vector<Node*>{n2, n0};
//     // std::cout << n2->get_value() << std::endl;
//     Node* n3 = g["add"](args);
//     // std::cout << n3->get_value() << std::endl;
//     // std::cout << g;

//     Node* n4 = new Node(10);
//     g.addNode(n4);

//     Node* n5 = g["sub"]( std::vector<Node*>{n3, n4} );

//     g.addNode(n0);
//     g.addNode(n1);
//     g.saveGraphToFile("graph.dot");
//     g.orderNodes();
//     g.backwards();
// }