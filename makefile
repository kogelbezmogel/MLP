CC = g++
CFLAGS = -g

main: main.o simple_tensor.o tensor.o tensor_operations.o utils.o node.o graph.o dataset.o dataloader.o layer.o model.o
	$(CC) $(CFLAGS) -o main.out main.o simple_tensor.o tensor.o tensor_operations.o utils.o node.o graph.o dataset.o dataloader.o layer.o model.o
	# $(MAKE) clean

main.o: src/main.cpp src/simple_tensor.h src/tensor.h src/utils.h src/node.h src/graph.h src/dataloader.h
	$(CC) $(CFLAGS) -c src/main.cpp

simple_tensor.o: src/simple_tensor.cpp src/simple_tensor.h
	$(CC) $(CFLAGS) -c src/simple_tensor.cpp

tensor.o: src/tensor.cpp src/simple_tensor.h src/graph.h
	$(CC) $(CFLAGS) -c src/tensor.cpp

tensor_operations.o: src/tensor_operations.cpp src/tensor_operations.h
	$(CC) $(CFLAGS) -c src/tensor_operations.cpp

utils.o: src/utils.cpp src/utils.h
	$(CC) $(CFLAGS) -c src/utils.cpp

node.o: src/node.cpp src/node.h src/simple_tensor.h
	$(CC) $(CFLAGS) -c src/node.cpp src/node.h src/simple_tensor.h

graph.o: src/graph.cpp src/graph.h src/node.h
	$(CC) $(CFLAGS) -c src/graph.cpp src/graph.h src/node.h

dataset.o: src/dataset.cpp
	$(CC) $(CFLAGS) -c src/dataset.cpp

dataloader.o: src/dataloader.cpp src/simple_tensor.h
	$(CC) $(CFLAGS) -c src/dataloader.cpp

layer.o: src/layer.cpp src/layer.h src/tensor.h
	$(CC) $(CFLAGS) -c src/layer.cpp src/layer.h src/tensor.h

model.o: src/model.cpp src/model.h src/layer.h src/tensor.h
	$(CC) $(CFLAGS) -c src/model.cpp src/model.h src/layer.h


clean: 
	rm -f *.o
	rm -f src/*.gch