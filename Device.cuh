#ifndef DEVICE_CUH_
#define DEVICE_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "curand_kernel.h"
#include "device_functions.h"

namespace Device {

	extern cublasHandle_t cublasHandle;

	void init();
	void reset();

	class Array {

		friend void clip(Array& c, Array& a, const float min, const float max);
		friend void expand(Array& c, Array& a);
		friend void scale(Array& c, Array& a, const float scalar);
		friend void collapseHighestDimension(Array& c, Array& a, Array& scale, size_t heighestDimensionSize);
		friend void absoluteValue(Array& c, Array& a);
		friend void randomizeNormal(Array& c, const float scale, const float offset);
		friend void average(Array& c, Array& a, Array& b);
		friend void hadamardProduct(Array& c, Array& a, Array& b);
		friend void addition(Array&c, Array& a, Array& b);
		friend void multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n);

		float *deviceData;
		size_t size;
		size_t bSize;

		float* getPtr(); // Friends should call this to access raw pointer.

	public:
		Array();
		Array(size_t size);
		Array(const Array& deviceArray);
		Array(Array&& deviceArray) noexcept;
		~Array();

		Array& operator=(const Array& deviceArray);
		Array& operator=(Array&& deviceArray) noexcept;

		void set(float* hostData);
		void get(float* hostData);

		size_t getSize();
	};

	// clip to range min-max
	void clip(Array& c, Array& a, const float min, const float max);
	__global__ void clipKernel(float* c, const float* a, const float min, const float max);

	// Add a new dimension onto array... copying current contents into new dimension units.
	void expand(Array& c, Array& a);
	__global__ void expandKernel(float* c, const float* a, size_t aSize);

	// Entrywise scale with given scalar
	void scale(Array& c, Array& a, const float scalar);
	__global__ void scaleKernel(float* c, const float* a, const float scalar);

	// summ and scale...
	void collapseHighestDimension(Array& c, Array& a, Array& scale, size_t highestDimensionSize);
	__global__ void collapseHeighestDimensionKernel(float* c, const float* a, const float* scale);

	// Absolute Value
	void absoluteValue(Array& c, Array& a);
	__global__ void absoluteValueKernel(float* c, const float* a);

	// Init Randomizer
	void randomInit(unsigned int states, unsigned long long seed);
	__global__ void randomInitKernel(unsigned long long seed, curandState_t* curandState);

	// Randomize all entries using normal distribution, sacling then offsetting...
	void randomizeNormal(Array& c, const float scale, const float offset);
	__global__ void randomizeNormalKernel(float* c, const float scale, const float offset, curandState_t* curandState);

	// Entrywise average
	// c <-- (a + b) / 2
	void average(Array& c, Array& a, Array& b);
	__global__ void averageKernel(float* c, const float* a, const float* b);

	// Entrywise product
	// c <-- a * b
	void hadamardProduct(Array& c, Array& a, Array& b);
	__global__ void hadamardProductKernel(float *c, const float* a, const float* b);


	// Entrywise addition
	// c <-- a + b
	void addition(Array&c, Array& a, Array& b);
	__global__ void additionKernel(float* c, const float* a, const float* b);

	// Matrix Multiplication
	// using cuBLAS.
	void multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n);
}

#endif // DEVICE_CUH_