echo OpenMP compile anaglyph:
g++ opencv-omp-anaglyph.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ opencv-omp-anaglyph.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-anaglyph

echo OpenMP compile image processing:
g++ opencv-omp-ip.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ opencv-omp-ip.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-ip
