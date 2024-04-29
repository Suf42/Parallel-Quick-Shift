This repo contains all the materials for the EE451 Spring 24 final project submitted by Taryn Sanders and Mohammad Sufian at the University of Southern California under professor Viktor Prasanna.<br><br>

<b>Introduction:</b><br>

Image processing has become a necessary discussion in various emerging industries for its ability to extract useful information from large quantities of image data. We plan to speed up image segmentation by parallelising the Quick Shift algorithm.<br>

Quick Shift is based on an approximation of Mean Shift Clustering. In its entirety, Quick Shift is a hierarchical clustering technique. It uses a non-parametric method which aims to identify homogeneous regions in an image based on similarities in color and spatial proximity by computing a density estimate at each pixel location.<br><br>

In this project, we aim to achieve the following tasks:

<ol>
<li>Implement a serial version of Quick Shift using C programming.</li>
<li>Implement a parallel version of Quick Shift using CUDA programming and shared memory.</li>
<li>Evaluate speedup on a GPU platform using different images of various complexities with our serial baseline.</li>
</ol><br>

We propose to use CUDA programming to copy the image to the GPU unit and break down the computation of the density of a pixel and its surrounding neighbors into multiple thread blocks.<br>

We also need to address memory access times since we need to access a lot of information about neighboring pixels. We propose to load a chunk of surrounding neighboring pixels into shared memory to promote data reuse. When a pixel in a block tries to compute similarity with a pixel outside of that block, the memory access can be cached to shared memory.<br><br>

Thus, our hypothesis is: By parallelizing the density calculation, the computational time and complexity  can be improved  by using shared memory provided by streaming multiprocessors and exploiting data reuse.<br><br>

<b>Content:</b><br>

This repo contains the following materials:<br>

<ol>
<li>README.md - refer to this for updated information on the project</li>

<li>report.pdf - this is the final report for our project</li>
<li>source folder - contains all the source codes for the project.</li>
</ol><br>

Inside the source folder, you will find main.cpp along with additional header files. Compile main.cpp with the latest version of G++ and your image named “test.png” in the same working directory. Run your compiled object code from your terminal.<br><br>

<b>Requirements:</b><br>

Since this project uses CUDA, a CUDA-enabled GPU with the latest version of the CUDA toolkit is required.
