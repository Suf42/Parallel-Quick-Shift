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
            dist[globalX * w + globalY] = ((x != globalX || y != globalY) ? sqrt(minDist) : INFINITY);
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
                    E += exp(-temp * 4.5 / sigma3 / sigma3);
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
    // compute pixel energy
    float *enrg = (float *)malloc(w * h * sizeof(float));
    getDist(w, h, d, sigma3, ratio, img, enrg);

    // compute distance
    getTree(w, h, d, sigma3, tau2, ratio, img, enrg, parent, dist);

    free(enrg);
}
