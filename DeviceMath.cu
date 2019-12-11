#include "Device.cuh"
#include <iostream>
#include <sstream>
#include "cublas_v2.h"

curandState_t* d_curandState = 0; //we can add a push-pop-swap function to swap between curandState arrays? ... maybe or just init with time(NULL) all da time.. :/

namespace Device {
	void checkError(std::string functionName, cudaError_t cudaError) {
		if (cudaError != cudaSuccess) {
			std::stringstream ss;
			ss << functionName << " failed: " << cudaGetErrorString(cudaError) << std::endl;
			throw std::runtime_error(ss.str().c_str());
		}
	}

	void clip(Array& c, Array& a, const float min, const float max) {
		if (c.getSize() == a.getSize()) {
			if (a.getSize() > 0) {
				clipKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), min, max);
				std::string functionName("Device::clip(Array& c, Array& a, const float min, const float max)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
		}
		else {
			throw std::runtime_error("Device::expandNewDimension(Array& c, Array& a, size_t newDimensionSize) failed: c size mismatch.");
		}
	}

	void expand(Array& c, Array& a) {
		if (a.getSize() > 0) {
			if(c.getSize() >= a.getSize() && c.getSize() % a.getSize() == 0) {
				expandKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), a.getSize());
				std::string functionName("Device::expand(Array& c, Array& a)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::expandNewDimension(Array& c, Array& a, size_t newDimensionSize) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::expandNewDimension(Array& c, Array& a, size_t newDimensionSize) failed: a must have size greater than 0.");
		}
	}

	void scale(Array& c, Array& a, const float scalar) {
		if (a.getSize() > 0) {
			if (a.getSize() == c.getSize()) {
				scaleKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), scalar);
				std::string functionName("Device::scale(Array& c, Array& a, const float scalar)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::scale(Array& c, Array& a, const float scalar) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::scale(Array& c, Array& a, const float scalar) failed: a must have size greater than 0.");
		}
	}

	void collapseHighestDimension(Array& c, Array& a, Array& scale, size_t highestDimensionSize) {
		if (highestDimensionSize == scale.getSize()) {
			if(a.getSize() / highestDimensionSize == c.getSize()) {
				Device::scale(c, c, 0.0f);
				dim3 threadsPerBlock((unsigned int)(a.getSize() / highestDimensionSize), highestDimensionSize);
				collapseHeighestDimensionKernel<<<1, threadsPerBlock>>>(c.getPtr(), a.getPtr(), scale.getPtr());
				std::string functionName("Device::collapseHighestDimension(Array& c, Array& a, Array& scale, size_t highestDimensionSize)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::scale(Array& c, Array& a, const float scalar) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::collapseHighestDimension(Array& c, Array& a, Array& scale, size_t highestDimensionSize) failed: scale size mismatch.");
		}

	}

	void absoluteValue(Array& c, Array& a) {
		if (c.getSize() > 0) {
			if (a.getSize() == c.getSize()) {
				std::string functionName("Device::absoluteValue(Array& c, Array& a)");
				absoluteValueKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr());
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::absoluteValue(Array& c, Array& a) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::absoluteValue(Array& c, Array& a) failed: c must have size greater than 0.");
		}
	}

	void randomInit(unsigned int states, unsigned long long seed) {
		std::string functionName("Device::randomizeNormal(Array& c, const float mean, const float standardDeviation, const float scale, const float offset)");
		cudaFree(d_curandState);
		cudaMalloc(&d_curandState, states * sizeof(curandState_t));
		checkError(functionName, cudaGetLastError());
		randomInitKernel<<<1, states>>>(seed, d_curandState);
		checkError(functionName, cudaGetLastError());
		checkError(functionName, cudaDeviceSynchronize());
	}

	void randomizeNormal(Array& c, const float scale, const float offset) {
		if (c.getSize() > 0) {
			randomizeNormalKernel<<<1, (unsigned int)c.getSize() >>>(c.getPtr(), scale, offset, d_curandState);
			std::string functionName("Device::randomizeNormal(Array& c, const float mean, const float standardDeviation, const float scale, const float offset)");
			checkError(functionName, cudaGetLastError());
			checkError(functionName, cudaDeviceSynchronize());
		}
		else {
			throw std::runtime_error("Device::randomizeNormal(Array& c, const float mean, const float standardDeviation, const float scale, const float offset) failed: c must have size greater than 0.");
		}
	}

	void hadamardProduct(Array& c, Array& a, Array& b) {
		if (a.getSize() == b.getSize()) {
			if (a.getSize() == c.getSize()) {
				hadamardProductKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), b.getPtr());
				std::string functionName("Device::hadamardProduct(Array&c, Array& a, Array& b)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::hadamardProduct(Array& c, Array& a, Array& b) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::hadamardProduct(Array&c, Array& a, Array& b) failed: a and b do not share the same dimensions.");
		}
	}

	void addition(Array& c, Array& a, Array& b) {
		if (a.getSize() == b.getSize()) {
			if (a.getSize() == c.getSize()) {
				additionKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), b.getPtr());
				std::string functionName("Device::addition(Array&c, Array& a, Array& b)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::addition(Array& c, Array& a, Array& b) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::addition(Array&c, Array& a, Array& b) failed: a and b must share the same dimensions.");
		}
	}

	void average(Array& c, Array& a, Array& b) {
		if (a.getSize() == b.getSize()) {
			if (a.getSize() == c.getSize()) {
				averageKernel<<<1, (unsigned int)c.getSize()>>>(c.getPtr(), a.getPtr(), b.getPtr());
				std::string functionName("Device::average(Array&c, Array& a, Array& b)");
				checkError(functionName, cudaGetLastError());
				checkError(functionName, cudaDeviceSynchronize());
			}
			else {
				throw std::runtime_error("Device::average(Array& c, Array& a, Array& b) failed: c size mismatch.");
			}
		}
		else {
			throw std::runtime_error("Device::average(Array& c, Array& a, Array& b) failed: a and b must share the same size.");
		}
	}

	void multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) {
		if (b.getSize() == k * n) {
			if (a.getSize() == m * k) {
				const float alf = 1;
				const float bet = 0;
				const float* alpha = &alf;
				const float* beta = &bet;
				cublasStatus_t status(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a.getPtr(), m, b.getPtr(), k, beta, c.getPtr(), m));
				switch (status) {
					case CUBLAS_STATUS_NOT_INITIALIZED: {
						throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed: Library not initialized.");
					}
					case CUBLAS_STATUS_INVALID_VALUE: {
						throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed: Parameters m, k, and n must not be less than 0.");
					}
					case CUBLAS_STATUS_ARCH_MISMATCH: {
						throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed: Architecture mismatch.");
					}
					case CUBLAS_STATUS_EXECUTION_FAILED: {
						throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed to launch on GPU.");
					}
				}
			}
			else {
				throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed: m*k must equal size of a.");
			}
			std::string functionName("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n)");
			checkError(functionName, cudaGetLastError());
			checkError(functionName, cudaDeviceSynchronize());
		}
		else {
			throw std::runtime_error("Device::multiply(Array& c, Array& a, Array& b, const int m, const int k, const int n) failed: k*n must equal size of b.");
		}
	}

	__global__ void clipKernel(float* c, const float* a, const float min, const float max) {
		int i = threadIdx.x;
		float t(min > a[i] ? min : a[i]);
		c[i] = t > max ? max : t;
	}

	__global__ void expandKernel(float* c, const float* a, size_t aSize) {
		int i = threadIdx.x;
		c[i] = a[i / aSize];
	}

	__global__ void scaleKernel(float* c, const float* a, const float scale) {
		int i = threadIdx.x;
		c[i] = a[i] * scale;
	}

	__global__ void collapseHeighestDimensionKernel(float* c, const float* a, const float* scale) {
		int i = threadIdx.x;
		int j = threadIdx.y;
		atomicAdd(&c[i], scale[j] * a[i + j * blockDim.x]);
	}

	__global__ void absoluteValueKernel(float* c, const float* a) {
		int i = threadIdx.x;
		float v(a[i]);
		c[i] = v < 0.0f ? -v : v;
	}

	__global__ void randomInitKernel(unsigned long long seed, curandState_t* d_curandState) {
		int i = threadIdx.x;
		curand_init(seed, i, 0, &d_curandState[i]);
	}

	__global__ void randomizeNormalKernel(float* c, const float scale, const float offset, curandState_t* d_curandState) {
		int i = threadIdx.x;
		c[i] = (curand_normal(&d_curandState[i]) * scale) + offset;
	}

	__global__ void averageKernel(float* c, const float* a, const float* b) {
		int i = threadIdx.x;
		c[i] = (a[i] + b[i]) / 2.0f;
	}

	__global__ void hadamardProductKernel(float* c, const float* a, const float* b) {
		int i = threadIdx.x;
		c[i] = a[i] * b[i];
	}

	__global__ void additionKernel(float* c, const float* a, const float* b) {
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}
}