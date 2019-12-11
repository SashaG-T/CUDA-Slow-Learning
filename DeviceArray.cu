#include "Device.cuh"
#include <iostream>

namespace Device {
	Array::Array()
		:
		deviceData(0),
		size(0),
		bSize(0)
	{}

	Array::Array(size_t size)
		:
		deviceData(0),
		size(size),
		bSize(size * sizeof(float))
	{
		cudaError_t cudaError(cudaMalloc(&deviceData, bSize));
		if (cudaError != cudaSuccess) {
			std::cout << "Error: Device::Array::DeviceArray(size_t size):" << std::endl << "\tFailed to allocate device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
		}

	}

	Array::Array(const Array& deviceArray)
		:
		deviceData(0),
		size(deviceArray.size),
		bSize(deviceArray.bSize)
	{
		cudaError_t cudaError(cudaMalloc(&deviceData, bSize));
		if (cudaError != cudaSuccess) {
			std::cout << "Error: Device::Array::DeviceArray(const DeviceArray& deviceArray):" << std::endl << "\tFailed to allocate device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
		}
		else {
			cudaError = cudaMemcpy((void *)this->deviceData, (void *)deviceArray.deviceData, bSize, cudaMemcpyDeviceToDevice);
			if (cudaError != cudaSuccess) {
				std::cout << "Error: Device::Array::DeviceArray(const DeviceArray& deviceArray):" << std::endl << "\tFailed to copy device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
			}
		}
	}

	Array::Array(Array&& deviceArray) noexcept
		:
		deviceData(deviceArray.deviceData),
		size(deviceArray.size),
		bSize(deviceArray.bSize)
	{
		deviceArray.deviceData = 0;
		deviceArray.size = 0;
		deviceArray.bSize = 0;
	}

	Array::~Array() {
		cudaError_t cudaError(cudaFree(deviceData));
		if (cudaError != cudaSuccess) {
			std::cout << "Error: Device::Array::~DeviceArray():" << std::endl << "\tFailed to free device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
		}
	};

	Array& Array::operator=(const Array& deviceArray) {
		if (this != &deviceArray) {
			cudaError_t cudaError(cudaFree(deviceData));
			if (cudaError != cudaSuccess) {
				std::cout << "Error: Device::Array::operator=(const DeviceArray& deviceArray):" << std::endl << "\tFailed to free device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
			}
			else {
				this->size = deviceArray.size;
				this->bSize = deviceArray.bSize;
				cudaError = cudaMalloc(&deviceData, bSize);
				if (cudaError != cudaSuccess) {
					std::cout << "Error: Device::Array::operator=(const DeviceArray& deviceArray):" << std::endl << "\tFailed to allocate device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
				}
				else {
					cudaError = cudaMemcpy((void *)this->deviceData, (void *)deviceArray.deviceData, bSize, cudaMemcpyDeviceToDevice);
					if (cudaError != cudaSuccess) {
						std::cout << "Error: Device::Array::operator=(const DeviceArray& deviceArray):" << std::endl << "\tFailed to copy device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
					}
				}
			}
		}
		return *this;
	}

	Array& Array::operator=(Array&& deviceArray) noexcept {
		if (this != &deviceArray) {
			this->size = deviceArray.size;
			this->bSize = deviceArray.bSize;
			this->deviceData = deviceArray.deviceData;
			deviceArray.deviceData = 0;
			deviceArray.size = 0;
			deviceArray.bSize = 0;
		}
		return *this;
	}

	float* Array::getPtr() {
		return deviceData;
	}

	void Array::set(float* hostData) {
		cudaError_t cudaError(cudaMemcpy((void *)deviceData, (void *)hostData, bSize, cudaMemcpyHostToDevice));
		if (cudaError != cudaSuccess) {
			std::cout << "Error: Device::Array::set(float* hostData):" << std::endl << "\tFailed to copy host memory to device memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
		}
	}

	void Array::get(float* hostData) {
		cudaError_t cudaError(cudaMemcpy((void *)hostData, (void *)deviceData, bSize, cudaMemcpyDeviceToHost));
		if (cudaError != cudaSuccess) {
			std::cout << "Error: Device::Array::get(float* hostData):" << std::endl << "\tFailed to copy device memory to host memory." << std::endl << "\tCudaError = " << cudaError << std::endl;
		}
	}

	size_t Array::getSize() {
		return size;
	}
}