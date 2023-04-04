### Practical Work 2: OpenMP and CUDA Based Image Processing
run from the directory cuda_and_openmp

``` sh compile.sh ```

if you want to run only for one method, you can see the commands for each method inside the compile.sh.

the implemented anaglyph methods equations were taken from this [link](https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx).

| Method Name | Method Number
| :--- | :----------
| True | 0
| Gray | 1
| Color | 2
| Half-Color | 3
| Optimized | 4


to execute run these: 

#### CUDA 

Anaglyph

```./imagecuda_a <<image path>> <<method number>>```

Example: ```./imagecuda_a flower_resized.jpg 2```

Gaussian 

```./imagecuda_ip <<image path>> <<kernel size divided by 2>> <<sigma>>```

Example: ```./imagecuda_ip flower_resized.jpg 3 1.0```

Denoising 

```./imagecuda_dn <<image path>> <<neighborhood size for the covariance matrix divided by 2>> <<factor ratio applied to determine the gaussian kernel size>>```

Example: ```./imagecuda_dn flower_resized.jpg 3 1.0```

#### OpenMP

Anaglyph

```./opencv-omp-anaglyph flower_resized.jpg 2```

Gaussian 

```./opencv-omp-processing flower_resized.jpg 3 1.0```

Denoising 

```./opencv-omp-denoising flower_resized.jpg 3 10000000.0```

