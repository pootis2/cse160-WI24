#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float *image, const float * __restrict__ mask, float *out, int channels, int width, int height){

    __shared__ float subInput[w][w][3];

    int tx = threadIdx.x; int bx = blockIdx.x;
    int ty = threadIdx.y; int by = blockIdx.y; 
    int tz = threadIdx.z;

    // output row col
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // input start row col
    int row1 = row - Mask_radius;
    int col1 = col - Mask_radius;

    if(row1 >= 0 && row1 < height && col1 >= 0 && col1 < width){
        subInput[ty][tx][tz] = image[(row1*width + col1)*3 + tz];
    }
    else{
        subInput[ty][tx][tz] = 0;
    }
    __syncthreads();
    float ans = 0;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH && row < height && col < width){
        for(int i = 0; i < Mask_width; i++){
            for(int j = 0; j < Mask_width; j++){
                ans += mask[i*Mask_width + j] * subInput[ty + i][tx + j][tz];
            }
        }
        out[(row * width + col)*3 + tz] = clamp(ans);
    }

}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskColumns * maskRows * sizeof(float));

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskColumns * maskRows * sizeof(float), cudaMemcpyHostToDevice);

  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimBlock(w, w, imageChannels); 
  dim3 dimGrid(ceil((float)imageWidth/TILE_WIDTH), ceil((float)imageHeight/TILE_WIDTH)); 
  convolution<<<dimGrid, dimBlock>>>(hostInputImageData, hostMaskData,
                                     hostOutputImageData, imageChannels,
                                     imageWidth, imageHeight);

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceMaskData);
  cudaFree(deviceOutputImageData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
