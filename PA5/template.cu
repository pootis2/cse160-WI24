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
#define w (TILE_WIDTH + Mask_width - 1) //number of loads from global memory i think(width of block)
#define clamp(x) (min(max((x), 0.0), 1.0)) // what is this used for?

//@@ INSERT CODE HERE
__global__ void convolution(float *deviceInputImageData, const float * __restrict__ deviceMaskData,
                            float* deviceOutputImageData, int imageChannels,
                            int imageWidth, int imageHeight){

    __shared__ float inputSharedMem[w][w]; //size is blockwidth x blockwidth, fit input into block (has the extra halo over the output)

      //blocks/threads in x and y direction
      int threadX = threadIdx.x;
      int threadY = threadIdx.y;

      int rowOutput = blockIdx.y * TILE_WIDTH + threadY; //row for output/tile
      int colOutput = blockIdx.x * TILE_WIDTH + threadX; //col for output/tile

      int rowInput = rowOutput - (Mask_width-1)/2; //row for input/block
      int colInput = colOutput - (Mask_width-1)/2; //col for input/block
      

    for(int i = 0; i < imageChannels; i++){
      //put input block into shared Mem
      if( (rowInput >= 0) && (rowInput < imageHeight) &&  (colInput >=0) && (colInput < imageWidth) ){
        inputSharedMem[threadY][threadX] = deviceInputImageData[(rowInput*imageWidth + colInput)*imageChannels + i]; //this would not put the first element in (0,0) of shared Mem
      }else{
        inputSharedMem[threadY][threadX] = 0;
      }

      //wait for all the shared mem to be filled
      __syncthreads();
      float outValue = 0;

      //now do the calculations for each tile  
      if(threadY < TILE_WIDTH && threadX < TILE_WIDTH)
      {
        for(int row = 0; row < Mask_width; row++){
          for(int col = 0; col < Mask_width; col++){
            outValue += deviceMaskData[row*Mask_width + col] * inputSharedMem[threadY + row][(threadX + col)];
          }
        }
      }
      __syncthreads();
      if(rowOutput < imageHeight && colOutput < imageWidth && threadY < TILE_WIDTH && threadX < TILE_WIDTH){
        deviceOutputImageData[(rowOutput * imageWidth + colOutput)*imageChannels + i] = clamp(outValue);
      }    
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
  cudaMalloc( (void**) &deviceInputImageData, sizeof(float)*imageWidth*imageHeight*imageChannels);
  cudaMalloc( (void**) &deviceOutputImageData, sizeof(float)*imageWidth*imageHeight*imageChannels);
  cudaMalloc( (void**) &deviceMaskData, sizeof(float)*maskRows*maskColumns);
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, sizeof(float)*imageChannels*imageWidth*imageHeight, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputImageData, hostOutputImageData, sizeof(float)*imageChannels*imageHeight*imageHeight, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, sizeof(float)*maskColumns*maskRows, cudaMemcpyHostToDevice);

  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimBlock(w, w, 1);

  dim3 dimGrid(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);

 convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                    deviceOutputImageData, imageChannels,
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
