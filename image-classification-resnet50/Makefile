
.PHONY: all clean

all: resnet50

resnet50: resnet50.cc
	$(CXX) -o $@ $^ -lmaccel

clean:
	rm resnet50
