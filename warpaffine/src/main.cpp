#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#define checkRuntime(op) __check_cuda_runtime((op),#op,__FILE__,__LINE__)

bool __check_cuda_runtime(cudaError_t code,const char* op,const char* file,int line){

    if(code!=cudaSuccess){

        const char* err_name=cudaGetErrorName(code);
        const char* err_message=cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed.\ncode=%s,message=%s.\n",file,line,op,err_name,err_message);
        return false;
    }
    return true;
}

//双线性插值
void warp_affine_bilinear(uint8_t* src,int src_width,int src_height,uint8_t* dst,int dst_width,int dst_height){


}

cv::Mat warpffine_to_center_align(cv::Mat image,const cv::Size& size){

    cv::Mat output(size,CV_8UC3);           //创建输出

    //创建两个指针，稍后会指向device上的内存
    uint8_t* psrc_device=nullptr;
    uint8_t* pdst_device=nullptr;

    size_t src_size=image.cols*image.rows;
    size_t dst_size=size.width*size.height;

    //在device上开辟空间
    checkRuntime(cudaMalloc(&psrc_device,src_size));
    checkRuntime(cudaMalloc(&pdst_device,dst_size));

    //将图像复制到device上
    checkRuntime(cudaMemcpy(psrc_device,image.data,src_size,cudaMemcpyHostToDevice));
    
    //使用双线性差值对图像进行缩放
    warp_affine_bilinear(psrc_device,image.cols,image.rows,pdst_device,size.width,size.height);

    //将device上的数据复制到host
    checkRuntime(cudaMemcpy(output.data,pdst_device,dst_size,cudaMemcpyDeviceToHost));

    //释放创建的指针
    checkRuntime(cudaFree(pdst_device));
    checkRuntime(cudaFree(psrc_device));

    return output;
}



int main(){

    cv::Mat image=cv::imread("car.jpg");
    cv::Mat output=warpffine_to_center_align(image,cv::Size(640,640));
    cv::imwrite("output.jpg",output);
    printf("Done.");

    return 0;
}