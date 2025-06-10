CXX=g++
CXXFLAGS=`pkg-config --cflags opencv4` -std=c++11
LDFLAGS=`pkg-config --libs opencv4`

all: fitness_app

fitness_app: fitness_app.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f fitness_app
