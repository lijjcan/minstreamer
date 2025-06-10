CXX=g++
CXXFLAGS=`pkg-config --cflags opencv4 Qt5Widgets` -std=c++11 -fPIC
LDFLAGS=`pkg-config --libs opencv4 Qt5Widgets`

all: fitness_app

fitness_app: fitness_app.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f fitness_app
