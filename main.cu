#include <iostream>
#include "Device.cuh"
#include <stdlib.h>

int main(void) {

	try {
		Device::init();
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
		std::exit(-1);
	}

	/* Orinal */ {

		//Attempt c = a*w;
		constexpr size_t size(8);
		Device::Array d_wArray(size*size);	//8x8 Prior Weights Matrix.
		Device::Array d_wpArray(size*size); //8x8 Post Weights Matrix.
		Device::Array d_aArray(size);		//1x8 Prior Activations Vector.
		Device::Array d_apArray(size);		//1x8 Post Activations Vector. (Result Vector)
		Device::Array d_wbArray(size*size); //8x8 Weight Bias (used to reward)
		Device::Array d_bArray(size);		//1x8 Random Bias (used to introduce randomness to the network)
		Device::Array d_rArray(size);		//1x8 Neuron Reward Trigger Strength (which neurons trigger reward and by how much)
		Device::Array d_arArray(size);		//1x8 hadamard(d_aArray, d_rArray): give us the current reward trigger strength for each neuron.
		Device::Array d_RArray(size*size);  //8x8 reward target for weights.
		Device::Array d_dArray(size*size*size); //an array that defines which neurons reward which weights.
		Device::Array d_cArray(size*size);	//8x8 expansion of activations for strengthen used weights and weaken unused ones.
		
		float* h_w(new float[size*size]);
		float* h_wp(new float[size*size]);
		float* h_a(new float[size]);
		float* h_ap(new float[size]);
		float* h_wb(new float[size*size]);
		float* h_b(new float[size]);
		float* h_r(new float[size]);
		float* h_d(new float[size*size*size]);

		for (int i = 0; i < size*size; i++) {
			h_wb[i] = 1.0f; //All weights will converge towards 1 when rewarded.
		}
		d_wbArray.set(h_wb);

		Device::randomInit(size*size*size, 12532);
		Device::randomizeNormal(d_dArray, 0.05f, 0.0f);
		Device::absoluteValue(d_dArray, d_dArray);

		d_dArray.get(h_d);
		for (int i = 0; i < size; i++) {
			h_d[i] = h_d[i] > 1.0f ? 1.0f : 0.0f;
		}
		d_dArray.set(h_d);

		Device::randomInit(size*size, 125);
		Device::randomizeNormal(d_wArray, 0.05f, 0.0f);
		Device::absoluteValue(d_wArray, d_wArray);

		Device::randomInit(size, 1534);
		Device::randomizeNormal(d_rArray, 0.05f, 0.0f);
		Device::absoluteValue(d_rArray, d_rArray);

		Device::randomizeNormal(d_aArray, 0.05f, 0.0f);
		Device::absoluteValue(d_aArray, d_aArray);

		d_rArray.get(h_r);
		for (int i = 0; i < size; i++) {
			h_r[i] = h_r[i] > 1.0f ? 1.0f : 0.0f;
		}
		d_rArray.set(h_r);

		float threshold(0.05f); //for visualization.
		for (int e = 0; e < 1000; e++) {
			for (int s = 0; s < 1000; s++) {
				//ap = aw
				Device::multiply(d_apArray, d_aArray, d_wArray, 1, size, size);

				//randmize b
				Device::randomizeNormal(d_bArray, 0.109375f, 0.0f); //0.10625

				//ap = ap + b;
				Device::addition(d_apArray, d_apArray, d_bArray);

				//ap = min(max(0, ap), 1)
				Device::clip(d_apArray, d_apArray, 0, 1);

				//ar = hadamard(a, r)
				Device::hadamardProduct(d_arArray, d_aArray, d_rArray);

				//R = sum(i = 0 to n, ar[i] * delta[i]) + d (d -> 0 matrix...)
				Device::collapseHighestDimension(d_RArray, d_dArray, d_arArray, size);

				//wp = (w + R) / 2
				Device::average(d_wpArray, d_wArray, d_RArray);

				//c = [x_0, x_1, x_2, ..., x_n], where x_i = a
				Device::expand(d_cArray, d_aArray);

				//wp = (wp + c) / 2
				Device::average(d_wpArray, d_wpArray, d_cArray);

				//scale weights down to simulate neurons getting tired and also neuron degredation...
				Device::scale(d_wpArray, d_wpArray, 0.9f); 

				/*
				d_aArray.get(h_a);
				d_apArray.get(h_ap);
				d_wArray.get(h_w);
				d_wpArray.get(h_wp);
				d_bArray.get(h_b);

				puts("Activations");
				for (int i = 0; i < size; i++) {
					std::cout << h_a[i] << " : " << h_b[i] << std::endl;
				}
				puts("Weights");
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						std::cout << h_w[i + j * size] << ", ";
					}
					std::cout << std::endl;
				}
				puts("Result");
				for (int i = 0; i < size; i++) {
					std::cout << h_ap[i] << std::endl;
				}
				*/
				//w = wp;
				d_wArray = d_wpArray;

				//a = ap;
				d_aArray = d_apArray;
			}

			d_aArray.get(h_a);
			int counter(0);
			for (int i = 0; i < size; i++) {
				std::cout << (h_a[i] > threshold ? (counter++, "+") : "-");
			}
			if (counter == size) {
				threshold += 0.01f;
				std::cout << "\t(+) Threshold = " << threshold;
			}
			if (counter == 0) {
				threshold -= 0.01f;
				std::cout << "\t(-) Threshold = " << threshold;
			}
			std::cout << std::endl;
		}

		d_aArray.get(h_a);
		d_apArray.get(h_ap);
		d_wArray.get(h_w);
		d_wpArray.get(h_wp);
		d_bArray.get(h_b);

		puts("Activations");
		for (int i = 0; i < size; i++) {
			std::cout << h_a[i] << " : " << h_b[i] << std::endl;
		}
		puts("Weights");
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				std::cout << h_w[i + j * size] << ", ";
			}
			std::cout << std::endl;
		}
		puts("Result");
		for (int i = 0; i < size; i++) {
			std::cout << h_ap[i] << std::endl;
		}

		delete[] h_w;
		delete[] h_wp;
		delete[] h_a;
		delete[] h_ap;
		delete[] h_wb;
		delete[] h_b;
		delete[] h_r;
		delete[] h_d;
	}
	
	puts("End of Processing.");
	try {
		Device::reset(); //must be called after Device::Array's are out of scope (hence the forced scope above)
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
		exit(-2);
	}

	return 0;
}