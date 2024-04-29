#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INF 9999999

void getTree(
    int w, int h, int d,
    float sigma3, float tau2, float ratio,
    float* img,
    float* enrg,
    float* parent,
    float* dist
){
    int globalX, globalY, x_start, x_end, y_start, y_end, i, j, k, x, y;
    float currE, minDist, atom, temp;
    float c[3];

    for (globalX = 0; globalX < h; ++globalX) {
        for (globalY = 0; globalY < w; ++globalY) {
            // pos of pixels to be checked
            x_start = globalX < sigma3 ? 0 : globalX - sigma3;
            x_end = globalX + sigma3 > w ? w : globalX + sigma3;
            y_start = globalY < sigma3 ? 0 : globalY - sigma3;
            y_end = globalY + sigma3 > h ? h : globalY + sigma3;

            // get the vector for current pixel
            for (k = 0; k < d; ++k) {
                c[k] = img[globalX * w * d + globalY * d + k];
            }

            // init temp variables
            currE = enrg[globalX * w + globalY];
            minDist = INFINITY;
            x = globalX;
            y = globalY;

            for (i = x_start; i < x_end; ++i) {
                for (j = y_start; j < y_end; ++j) {
                    if (enrg[i * w + j] > currE) {
                        temp = 0;
                        for (k = 0; k < d; ++k) {
                            atom = img[i * w * d + j * d + k] - c[k];
                            temp += atom * atom;
                        }
                        atom = ratio * (globalX - i);
                        temp += atom * atom;
                        atom = ratio * (globalY - j);
                        temp += atom * atom;
                        if (temp < minDist && temp < tau2) {
                            x = i;
                            y = j;
                            minDist = temp;
                        }
                    }
                }
            }

            parent[globalX * w + globalY] = (x * w + y); // parent[.] % width = y
            dist[globalX * w + globalY] = ((x != globalX || y != globalY) ? sqrtf(minDist) : INFINITY);
        }
    }
}

void getDist(
    int w, int h, int d,
    float sigma3, float ratio,
    float* img,
    float* enrg
){
    int globalX, globalY, x_start, x_end, y_start, y_end, i, j, k;
    float E, temp, atom;
    float c[3];

    for (globalX = 0; globalX < h; ++globalX) {
        for (globalY = 0; globalY < w; ++globalY) {
            // pos of pixels to be checked
            x_start = globalX < sigma3 ? 0 : globalX - sigma3;
            x_end = globalX + sigma3 > w ? w : globalX + sigma3;
            y_start = globalY < sigma3 ? 0 : globalY - sigma3;
            y_end = globalY + sigma3 > h ? h : globalY + sigma3;

            // cache the vector for current pixel
            for (k = 0; k < d; ++k) {
                c[k] = img[globalX * w * d + globalY * d + k];
            }

            // init temp variables
            E = 0;

            for (i = x_start; i < x_end; ++i) {
                for (j = y_start; j < y_end; ++j) {
                    temp = 0;
                    // get distance
                    for (k = 0; k < d; ++k) {
                        atom = img[i * w * d + j * d + k] - c[k];
                        temp += atom * atom;
                    }
                    atom = ratio * (globalX - i);
                    temp += atom * atom;
                    atom = ratio * (globalY - j);
                    temp += atom * atom;
                    // add into energy
                    E += expf(-temp * 4.5 / sigma3 / sigma3);
                }
            }
            // normalize
            enrg[globalX * w + globalY] = E / (x_end - x_start) / (y_end - y_start);
        }
    }
}

void quickshift(
    float* img,
    int w, int h, int d,
    float sigma3, float tau2, float ratio,
    float* parent, float* dist
){
    // generate texture for image
    cudaArray* arr_img;
    cudaChannelFormatDesc des_img = cudaCreateChannelDesc<float>();
    cudaExtent const ext = {w, h, d};
    cudaMalloc3DArray(&arr_img, &des_img, ext);

    cudaMemcpy3DParms cpyParms = {0};
    cpyParms.dstArray = arr_img;
    cpyParms.kind = cudaMemcpyHostToDevice;
    cpyParms.extent = make_cudaExtent(w, h, d);
    cpyParms.srcPtr = make_cudaPitchedPtr((void*) &img[0], ext.width * sizeof(float), ext.width, ext.height);
    cudaMemcpy3D(&cpyParms);

    cudaBindTextureToArray(texImg, arr_img, des_img);
    texImg.normalized = false;
    texImg.filterMode = cudaFilterModePoint;

    // compute pixel energy
    float *img_d, *dist_d, *parent_d, *enrg_d;
    size_t size = w * h * sizeof(float);
    cudaMalloc((void**) &img_d, size * d);
    cudaMalloc((void**) &dist_d, size);
    cudaMalloc((void**) &parent_d, size);
    cudaMalloc((void**) &enrg_d, size);
    cudaMemcpy(img_d, img, size * d, cudaMemcpyHostToDevice);
    cudaMemset(enrg_d, 0, size);

    dim3 dimBlock(B_X, B_Y, 1);
    dim3 dimGrid((h % B_X != 0) ? h / B_X + 1 : h / B_X, (w % B_Y != 0) ? w / B_Y + 1 : w / B_Y, 1);
    getDist<<<dimGrid, dimBlock>>>(w, h, d, sigma3, ratio, img_d, enrg_d);
    cudaDeviceSynchronize();

    // generate texture for energy
    cudaArray* arr_enrg;
    cudaChannelFormatDesc des_enrg = cudaCreateChannelDesc<float>();
    cudaMallocArray(&arr_enrg, &des_enrg, w, h);
    size_t size_enrg = w * h * sizeof(float);
    cudaMemcpyToArray(arr_enrg, 0, 0, enrg_d, size_enrg, cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(texEnrg, arr_enrg, des_enrg);
    texEnrg.normalized = false;
    texEnrg.filterMode = cudaFilterModePoint;

    // compute distance
    getTree<<<dimGrid, dimBlock>>>(w, h, d, sqrtf(tau2), tau2, ratio, parent_d, dist_d);
    cudaDeviceSynchronize();

    // copy parent & distance from device to host
    cudaMemcpy(parent, parent_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dist, dist_d, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(img_d);
    cudaFree(dist_d);
    cudaFree(parent_d);
    cudaFree(enrg_d);
    cudaFreeArray(arr_img);
    cudaFreeArray(arr_enrg);
    cudaUnbindTexture(texImg);
    cudaUnbindTexture(texEnrg);
}

__global__ void getDist_shared(
    int w, int h, int sigma3, float ratio2,
    float* img_d, float* enrg_d
){
    extern __shared__ float color[];
    int globalX = blockIdx.x * blockDim.x + threadIdx.x - sigma3;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y - sigma3;
    if(globalX < 0 || globalY < 0 || globalX >= h || globalY >= w) { return; }

    color[threadIdx.x * blockDim.y + threadIdx.y] =
        img_d[globalX * w + globalY + w * h * blockIdx.z];
    __syncthreads();

    float center = color[sigma3 * blockDim.y + sigma3];
    float d = color[threadIdx.x * blockDim.y + threadIdx.y] - center;
    d = d * d;
    if(blockIdx.z == 2)
    {
        d += ratio2 * (sigma3 - threadIdx.x) * (sigma3 - threadIdx.x);
    }
    else if(blockIdx.z == 1)
    {
        d += ratio2 * (sigma3 - threadIdx.y) * (sigma3 - threadIdx.y);
    }

    atomicAdd(enrg_d + blockIdx.x * w + blockIdx.y, d);
}

__global__ void getTree_shared(
    float* dist_d,
    int w, int h, int sigma3, float tau2,
    float* parent_d
){
    extern __shared__ float enrg[];
    int globalX = blockIdx.x * blockDim.x + threadIdx.x - sigma3;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y - sigma3;
    if(globalX < 0 || globalY < 0 || globalX >= h || globalY >= w) { return; }
    if(globalX == 0 && globalY == 0)
    {
        enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)] = INF;
    }
    enrg[threadIdx.x * blockDim.y + threadIdx.y] = exp(-dist_d[globalX * w + globalY] *
        4.5 / sigma3 / sigma3);
    int x_start = globalX > sigma3 ? globalX - sigma3 : 0;
    int x_end = globalX + sigma3 < h ? globalX + sigma3 : h;
    int y_start = globalY > sigma3 ? globalY - sigma3 : 0;
    int y_end = globalY + sigma3 < w ? globalY + sigma3 : w;
    enrg[threadIdx.x * blockDim.y + threadIdx.y] /= (x_end - x_start) * (y_end - y_start);
    __syncthreads();

    float center = enrg[sigma3 * blockDim.y + sigma3];
    if(enrg[threadIdx.x * blockDim.y + threadIdx.y] < center){ return; }
    if(enrg[threadIdx.x * blockDim.y + threadIdx.y] >
        exp(-tau2 * 4.5 / sigma3 / sigma3) / (x_end - x_start) / (y_end - y_start))
    { return; }

    if(enrg[threadIdx.x * blockDim.y + threadIdx.y] <
       enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)])
    {
        enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)] =
            enrg[threadIdx.x * blockDim.y + threadIdx.y];
        parent_d[blockIdx.x * w + blockIdx.y] = globalX * w + globalY;
    }
}

void quickshift_shared(
    float* img,
    int w, int h, int d,
    float sigma3, float tau2, float ratio,
    float* parent
) {
    if (sigma3 > 54) {
        printf("No support for large sigma(>18)!\nEnd Program.");
        return;
    }

    float *img_d, *dist_d, *parent_d;

    cudaMalloc((void**)&img_d, w * h * d * sizeof(float));
    cudaMemcpy(img_d, img, w * h * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dist_d, w * h * sizeof(float));
    cudaMemset(dist_d, 0, w * h * sizeof(float));

    cudaMalloc((void**)&parent_d, w * h * sizeof(float));

    dim3 dimGrid(w, h, 3);
    size_t shared = 2 * (int)sigma3 + 1;
    dim3 dimBlock(shared, shared, 1);
    getDist_shared<<<dimGrid, dimBlock, shared * shared * sizeof(float)>>>(
        w, h, (int)sigma3, ratio * ratio, dist_d);
    cudaDeviceSynchronize();

    dim3 dimGrid2(w, h, 1);
    getTree_shared<<<dimGrid2, dimBlock, (shared * shared + 1) * sizeof(float)>>>(
        dist_d, w, h, (int)sigma3, tau2, parent_d);
    cudaDeviceSynchronize();

    cudaMemcpy(parent, parent_d, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(img_d);
    cudaFree(dist_d);
    cudaFree(parent_d);
}



