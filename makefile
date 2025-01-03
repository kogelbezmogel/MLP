CC = g++
CFLAGS = -g

main: main.o simple_tensor.o tensor.o utils.o node.o graph.o dataloader.o layer.o model.o
	$(CC) $(CFLAGS) -o main main.o simple_tensor.o tensor.o utils.o node.o graph.o dataloader.o layer.o model.o
	$(MAKE) clean

main.o: main.cpp simple_tensor.h tensor.h utils.h node.h graph.h dataloader.h
	$(CC) $(CFLAGS) -c main.cpp

simple_tensor.o: simple_tensor.cpp simple_tensor.h
	$(CC) $(CFLAGS) -c simple_tensor.cpp

tensor.o: tensor.cpp simple_tensor.h graph.h
	$(CC) $(CFLAGS) -c tensor.cpp

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) -c utils.cpp

node.o: node.cpp node.h simple_tensor.h
	$(CC) $(CFLAGS) -c node.cpp node.h simple_tensor.h

graph.o: graph.cpp graph.h node.h
	$(CC) $(CFLAGS) -c graph.cpp graph.h node.h

dataloader.o: dataloader.cpp simple_tensor.h
	$(CC) $(CFLAGS) -c dataloader.cpp

layer.o: layer.cpp layer.h tensor.h
	$(CC) $(CFLAGS) -c layer.cpp layer.h tensor.h

model.o: model.cpp model.h layer.h tensor.h
	$(CC) $(CFLAGS) -c model.cpp model.h layer.h


clean: 
	rm -f *.o
	rm -f *.gch