echo CUDA compile image processing:

/usr/local/cuda/bin/nvcc image_processing.cu `pkg-config opencv4 --cflags --libs` imagecpp_image_processing.cpp -o imagecuda_ip

echo CUDA compile Anaglyph:

/usr/local/cuda/bin/nvcc image_anaglyph.cu `pkg-config opencv4 --cflags --libs` imagecpp_anaglyph.cpp -o imagecuda_a
