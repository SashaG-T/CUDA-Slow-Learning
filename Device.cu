#include "Device.cuh"
#include <iostream>

namespace Device{

	cublasHandle_t cublasHandle;

	void init() {
		cudaError_t cudaError(cudaSetDevice(0));
		if (cudaError != cudaSuccess) {
			throw std::runtime_error("Device::init() failed to initialize device.");
		}
		cublasStatus_t status(cublasCreate(&cublasHandle));
		if (status != CUBLAS_STATUS_SUCCESS) {
			if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
				throw std::runtime_error("Device::init() failed to initialize cuBLAS.");
			}
			else if(status == CUBLAS_STATUS_ALLOC_FAILED) {
				throw std::runtime_error("Device::init() failed to allocate resources for cuBLAS.");
			}
		}
	}
	void reset() {
		cublasStatus_t status(cublasDestroy(cublasHandle));
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			throw std::runtime_error("Device::init() failed to reset device? (cuBLAS library was not initialized)");
		}
		cudaError_t cudaError(cudaDeviceReset());
		if (cudaError != cudaSuccess) {
			throw std::runtime_error("Device::reset() failed to reset device.");
		}
	}
}