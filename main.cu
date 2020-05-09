#include "png++/png.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <thrust/complex.h>
typedef float float_; /*useful for testing float vs double*/
typedef unsigned int heatmap;

const int width = 500, height = 500;
const unsigned long long samples = 1000;
const unsigned long long sampleSamples = 10000;

/*map function*/
template <typename T>
__device__ constexpr T map(T x, T in_min, T in_max, T out_min, T out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ bool good(thrust::complex<float_> c) {
    // test H1 and H2

    float_ c2 = norm(c);
    thrust::complex<float_> d = c + thrust::complex<float_>(1.0, 0.0);
    float_ h1 = 256.0 * c2 * c2 - 96.0 * c2 + 32.0 * c.real() - 3.0;
    float_ h2 = 16.0 * (norm(d)) - 1.0;
    if (h1 > 0.0 && h2 > 0.0)
        return false;
    return true;
};

/*mandelbrot set function*/
__device__ void mSet(thrust::complex<float_> c, thrust::complex<float_>* Set,
    int* iterations, int maxIterations) {
    /*if point is the main cardioid it will not escape*/
    if (good(c)) {
        *iterations = 0;
        return;
    }

    auto z = thrust::complex<float_>(0, 0);

    *iterations = 0;
    while (norm(z) <= 5 && *iterations < maxIterations) {

        z = z * z + c;
        /*keep track of the orbit of z*/
        Set[*iterations] = z;
        ++(*iterations);
    }
    /*if iterations is 0(the point did not escape) discard the information*/
    if (*iterations == maxIterations)
        *iterations = 0;
}

__global__ void generateSamples(thrust::complex<float_>* Set, int* iterations,
    int maxIterations, thrust::complex<float_> minr,
    thrust::complex<float_> mini) {
    /*create a random number generator for both the real values and imaginary
     * values*/
    curandStateMRG32k3a_t realRand, imagRand;
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    static unsigned long long seed = 0;
    for (; index < samples; index += stride) {
        // seed a random number generator
        seed += index;
        curand_init(seed, 0, 0, &realRand);
        curand_init(curand(&realRand), 0, 0, &imagRand);
        thrust::complex<float_> c(
            map<float_t>(curand_uniform(&realRand), 0, 1, minr.real(), minr.imag()),
            map<float_t>(curand_uniform(&imagRand), 0, 1, mini.real(),
                mini.imag()));
        /*generate points*/
        mSet(c, Set + index * (unsigned long long)maxIterations, iterations + index,
            maxIterations);
    }
}

__global__ void addToHeatmap(heatmap* buffer, thrust::complex<float_>* Set,
    int* iterations, int* maxIterations,
    heatmap* maxValues, thrust::complex<float_> minr,
    thrust::complex<float_> mini) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    for (; index < samples * maxIterations[3]; index += stride) {
        int currentIteration = (index % maxIterations[3]);
        if (currentIteration >= iterations[index / maxIterations[3]]) {
            continue;
        }
        float_ real = Set[index].real(), imag = Set[index].imag();
        if (real < minr.real() || real > minr.imag() || imag < mini.real() ||
            imag > mini.imag())
            continue; // if point is out of bounds continue

        int row = map<float_>(real, minr.real(), minr.imag(), 0, (float_)width - 1);
        int col =
            map<float_>(imag, mini.real(), mini.imag(), 0, (float_)height - 1);
        int pixelIndex = row * width + col;
        /*red*/
        if (currentIteration < maxIterations[0]) {
            buffer[pixelIndex]++;
            if (buffer[pixelIndex] > maxValues[0]) {
                maxValues[0] = buffer[pixelIndex];
            }
        }
        /*green*/
        if (currentIteration < maxIterations[1]) {
            buffer[pixelIndex + width * height]++;
            if (buffer[pixelIndex + width * height] > maxValues[1]) {
                maxValues[1] = buffer[pixelIndex + width * height];
            }
        }
        /*blue*/
        if (currentIteration < maxIterations[2]) {
            buffer[pixelIndex + width * height * 2]++;
            if (buffer[pixelIndex + width * height * 2] > maxValues[2]) {
                maxValues[2] = buffer[pixelIndex + width * height * 2];
            }
        }
    }
}

int getColor(heatmap value, heatmap maxValue) {
    double scl = ((double)value) / ((double)maxValue);
    scl = scl > 1 ? 1 : scl; // clamp scl
    return round(scl * 255.0);
}

int main() {
    /*mapping*/
    thrust::complex<float_> minr = thrust::complex<float_>(-2.0, .828);
    thrust::complex<float_> mini = thrust::complex<float_>(-1.414, 1.414);

    auto t1 = std::chrono::high_resolution_clock::now();

    int* iter;
    cudaMallocManaged(&iter, sizeof(int) * 4);
    /*red,        green,        and blue*/
    iter[0] = 800, iter[1] = 500, iter[2] = 50;
    /*find the highest max iteration*/
    iter[3] = std::max({ iter[0], iter[1], iter[2] });
    /*allocate memory for heatmap buffer*/
    heatmap* buffer;
    /*allocate memory for storing the highest values in the buffer*/
    heatmap* maxValues;
    cudaMallocManaged(&buffer, sizeof(heatmap) * width * height * 3);
    cudaMallocManaged(&maxValues, sizeof(heatmap) * 3);
    /*allocate memory for storing the orbits of points that escape*/
    thrust::complex<float_>* Set;
    /*allocate memory for stroring the number of iterations a point takes to
     * escape*/
    int* iterations;
    cudaMallocManaged(&Set, sizeof(thrust::complex<float_>) * samples * iter[3]);
    cudaMallocManaged(&iterations, sizeof(int) * samples);

    /*sample multiple times*/
    for (int i = 0; i < sampleSamples; ++i) {
        generateSamples << <samples, 1024 >> > (Set, iterations, iter[3], minr, mini);
        cudaDeviceSynchronize();
        addToHeatmap << <32, 1024 >> > (buffer, Set, iterations, iter, maxValues, minr,
            mini);
        cudaDeviceSynchronize();
    }

    /*output image*/
    png::image<png::rgb_pixel> image(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            int index = j * height + i; // rotate 90 degrees
            image.set_pixel(
                i, j,
                png::rgb_pixel(
                    getColor(buffer[index], maxValues[0]),
                    getColor(buffer[index + width * height], maxValues[1]),
                    getColor(buffer[index + width * height * 2], maxValues[2])));
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    double timeTook = duration.count();
    std::cout << "It took " << timeTook
        << ((timeTook == 1.0) ? " second" : "seconds") << "\n";

    image.write("output.png");
    /*free variables*/
    cudaFree(iter);
    cudaFree(buffer);
    cudaFree(iterations);
    cudaFree(Set);
    cudaFree(maxValues);
}