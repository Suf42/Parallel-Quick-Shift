#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"
#define INF FLT_MAX

// #define MIN_PIXELS 20
// #include <algorithm>
// #include <climits>

// g++ -o main main.cpp
// ./main -h

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
	return std::find(begin, end, option) != end;
}

char* getCmdOption(char** begin, char** end, const std::string& option) {
	char** it = std::find(begin, end, option);
	if(it != end && ++it != end)
	{
		return *it;
	}
	return 0;
}

void getImg(char* name, float* img, int width, int height, int depth) {
	unsigned char* image_char = (unsigned char*) malloc(width * height * depth * sizeof(unsigned char));
	float * offset_float;
	for(int i = 0; i < height; ++i) {
		for(int j = 0; j < width; ++j) {
			offset_float = img + i * width + j;
			for(int k = 0; k < depth; ++k) {
				image_char[i * width * depth + j * depth + k] = (char)* (offset_float + k * width * height);
			}
		}
	}
	stbi_write_png(name, width, height, depth, image_char, width * depth * sizeof(unsigned char));

}

void getOutput(float* img, float* data, int width, int height, int depth, bool gpu, int s, int t, int r)
{
	// get the flat graph
	int size = width * height;
	int *map = (int*) malloc(size * sizeof(int));
	int num = 0;
	for(int i = 0; i < size; ++i) {
		map[i] = data[i];
	}

	bool flag = true;
	while(flag) {
		flag = false;
		for(int i = 0; i < size; ++i) {
			flag = flag || (map[i] != map[map[i]]);
			map[i] = map[map[i]];
		}
	}

	// get mean RGB values of each tree
	float* mean = (float*) calloc(size * depth, sizeof(float));
	float* count = (float*) calloc(size, sizeof(float));
	for(int i = 0; i < size; ++i) {
		count[map[i]]++;
		for(int k = 0; k < depth; ++k) {
			mean[map[i] + k * width * height] += img[i + k * width * height];
		}
	}

	for(int i = 0 ; i < size; ++i) {
		for(int k = 0; k < depth; ++k) {
			mean[i + k * width * height] /= count[i];
		}
	}
	free(count);
	
	// color segments
	for(int i = 0; i < size; ++i) {
		for(int k = 0; k < depth; ++k) {
			mean[i + k * width * height] = mean[map[i] + k * width * height];
		}
	}

	// display superpixel boundary
	for(int i = 0; i < size; ++i) {
		if(map[i] == i) {
			num++;
		}
	}
	printf("num of superpixel is %d.\n", num);

	if(gpu){
		getImg("test_gpu.png", mean, width, height, depth);
	}
	else{
		getImg("test_cpu.png", mean, width, height, depth);
	}

	// free(count);
	free(map);
}

void serialquickshift(float* img, float* enrg, int width, int height, int depth, int sigmaTimesThree, int tauSquared, int ratio, float* parent, float* dist) {
	int x_start;
	int y_start;
	int x_end;
	int y_end;
	float temp;
	float atom;
	float E;
	// get enrg for each pixel
	for(int i = 0; i < height; ++i) {
		for(int j = 0; j < width; ++j) {
			E = 0;
			x_start = i < sigmaTimesThree ? 0 : i - sigmaTimesThree;
			x_end = i + sigmaTimesThree > height ? height : i + sigmaTimesThree;
			y_start = j < sigmaTimesThree ? 0 : j - sigmaTimesThree;
			y_end = j + sigmaTimesThree > width ? width : j + sigmaTimesThree;
			for(int m = x_start; m < x_end; ++m) {
				for(int n = y_start; n < y_end; ++n) {
					temp = 0;
					// get distance
					for(int k = 0; k < depth; ++k) {
						atom =  img[m * width + n + k * width * height] -
										img[i * width + j + k * width * height];
						temp += atom * atom;
					}
					atom = ratio * (m - i);
					temp += atom * atom;
					atom = ratio * (n - j);
					temp += atom * atom;
					E += exp(-temp * 4.5 / sigmaTimesThree / sigmaTimesThree);
				}
			}
			enrg[i * width + j] = E / (x_end - x_start) / (y_end - y_start);
		}
	}

	int x;
	int y;
	// get dist for each pixel
	for(int i = 0; i < height; ++i) {
		for(int j = 0; j < width; ++j) {
			E = enrg[i * width + j];
			dist[i * width + j] = INF;
			x = i;
			y = j;
			x_start = i < sigmaTimesThree ? 0 : i - sigmaTimesThree;
			x_end = i + sigmaTimesThree > height ? height : i + sigmaTimesThree;
			y_start = j < sigmaTimesThree ? 0 : j - sigmaTimesThree;
			y_end = j + sigmaTimesThree > width ? width : j + sigmaTimesThree;
			for(int m = x_start; m < x_end; ++m) {
				for(int n = y_start; n < y_end; ++n) {
					if(enrg[m * width + n] > E) {
						temp = 0;
						for(int k = 0; k < depth; ++k) {
							atom = img[m * width + n + k * width * height] -
											img[i * width + j + k * width * height];
							temp += atom * atom;
						}
						atom = ratio * (m - i);
						temp += atom * atom;
						atom = ratio * (n - j);
						temp += atom * atom;
						if(temp < dist[i * width + j] && temp < tauSquared) {
							dist[i * width + j] = temp;
							x = m;
							y = n;
						}
					}
				}
			}
			parent[i * width + j] = x * width + y;
			dist[i * width + j] = sqrt(dist[i * width + j]);
		}
	}
	return;
}

void quickshift_shared(float* img, int width, int height, int depth, int sigmaTimesThree, int tauSquared, int ratio, float* parent) {
	return;
}

int main(int argc, char **argv) {
    if(cmdOptionExists(argv, argv + argc, "-h") || cmdOptionExists(argv, argv + argc, "--h")) {
		printf("\nThis program implements quickshift for image segmentation.\n");
		printf("Only supports PNG images.\n\n");
		printf("Default values:\n");
		printf("sigma: 6\n");
		printf("tau: 10\n");
		printf("ratio: 1\n\n");
		printf("You may change your parameters with '--sigma', '--tau', --ratio'.\n\n");
	}

	float sigma = 6;
	float tau = 10;
	float ratio = 1;
	char* input = "test4.png.jpeg";

	if(cmdOptionExists(argv, argv + argc, "--sigma"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--sigma");
		std::string str(temp);
		sigma = std::stof(str);
	}

	if(cmdOptionExists(argv, argv + argc, "--tau"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--tau");
		std::string str(temp);
		tau = std::stof(str);
	}

	if(cmdOptionExists(argv, argv + argc, "--ratio"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--ratio");
		std::string str(temp);
		ratio = std::stof(str);
	}

	printf("Input arguments are: ratio %f, sigma %f, tau %f.\n\n", ratio, sigma, tau);
	float tauSquared = tau * tau;
	float sigmaTimesThree = 3 * sigma;

	// load image
	int width;
	int height;
	int numComponents;
	int depth = 3;
	unsigned char * image = stbi_load(input, &width, &height, &numComponents, 3);
	size_t size = width * height;
	float* img;
	unsigned char* offset;
	img = (float*) malloc(size * depth * sizeof(float));
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			offset = image + (i * width + j) * depth;
			for(int k = 0; k < depth; k++) {
				img[k * width * height + i * width + j] = (float) (offset[k] - 0);
			}
		}
	}
	printf("image size: width %d height %d\n", width, height);

	float *parent, *dist, *enrg;
	parent = (float*) calloc(size, sizeof(float));
	dist = (float*) calloc(size, sizeof(float));
	enrg = (float*) calloc(size, sizeof(float));

	// char* output = "input_check.png";
	// getImg(output, img, width, height, depth);

	// ------------------------- gpu -------------------------

	struct timeval gpu1, gpu2;
	gettimeofday(&gpu1, 0);
	quickshift_shared(img, width, height, depth, sigmaTimesThree, tauSquared, ratio, parent);
	gettimeofday(&gpu2, 0);
	double gputime = (1000000.0 * (gpu2.tv_sec - gpu1.tv_sec) + gpu2.tv_usec - gpu1.tv_usec) / 1000.0;
	printf("img seg time of gpu is %f.\n", gputime);

	bool gpu = true;
	getOutput(img, parent, width, height, depth, gpu, sigma, tau, ratio);

	// quick shift with shared mem on gpu
	// float* red = (float*) malloc(size * sizeof(float));
	// float* green = (float*) malloc(size * sizeof(float));
	// float* blue = (float*) malloc(size * sizeof(float));
	// memcpy(img, red, size * sizeof(float));
	// memcpy(img + size, green, size * sizeof(float));
	// memcpy(img + 2 * size, blue, size * sizeof(float));

	// ------------------------- gpu -------------------------

	// ------------------------- cpu -------------------------

	struct timeval cpu1, cpu2;
	gettimeofday(&cpu1, 0);
	serialquickshift(img, enrg, width, height, depth, sigmaTimesThree, tauSquared, ratio, parent, dist);
	gettimeofday(&cpu2, 0);
	double cputime = (1000000.0 * (cpu2.tv_sec - cpu1.tv_sec) + cpu2.tv_usec - cpu1.tv_usec) / 1000.0;
	printf("img seg time of cpu is %f.\n", cputime);

	gpu = false;
	getOutput(img, parent, width, height, depth, gpu, sigma, tau, ratio);

	// ------------------------- cpu -------------------------

	printf("Terminate program.");
	free(parent);
	free(dist);
	free(enrg);
	free(img);
	free(image);
	return 0;
}