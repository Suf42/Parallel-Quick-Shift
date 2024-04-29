# Parallel-Quick-Shift
This repo contains all the materials for the EE451 Spring 24 final project submitted by Taryn Sanders and Mohammad Sufian.<br>

<b>Introduction:</b><br>

Image processing has become a necessary discussion in various emerging industries for its ability to extract useful information from large quantities of image data. We plan to speed up image segmentation by parallelising the Quick Shift algorithm.<br>

Quick Shift is based on an approximation of Mean Shift Clustering. In its entirety, Quick Shift is a hierarchical clustering technique. It uses a non-parametric method which aims to identify homogeneous regions in an image based on similarities in color and spatial proximity by computing a density estimate at each pixel location.<br>

In this project, we aim to achieve the following tasks:

<ol>
<li>Implement a serial version of Quick Shift using C programming.</li>
<li>Implement a parallel version of Quick Shift using CUDA programming and shared memory.</li>
<li>Evaluate speedup on a GPU platform using different images of various complexities with our serial baseline.</li>
</ol>

We propose to use CUDA programming to copy the image to the GPU unit and break down the computation of the density of a pixel and its surrounding neighbors into multiple thread blocks.<br>

We also need to address memory access times since we need to access a lot of information about neighboring pixels. We propose to load a chunk of surrounding neighboring pixels into shared memory to promote data reuse. When a pixel in a block tries to compute similarity with a pixel outside of that block, the memory access can be cached to shared memory.<br>

Thus, our hypothesis is: By parallelizing the density calculation, the computational time and complexity  can be improved  by using shared memory provided by streaming multiprocessors and exploiting data reuse.<br>

<b>Content:</b><br>
