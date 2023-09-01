#include <cuda_runtime.h>

#include <affine.cuh>

#include <math.h>

#include <opencv2/opencv.hpp>


//求解变换矩阵的结构体
struct AffineMatrix{

    float i2d[6];
    float d2i[6];

    void invertAffineTransform(float imat[6],float omat[6]){

        float i00=imat[0];float i01=imat[1]; float i02=imat[2];
        float i10=imat[3];float i11=imat[4]; float i12=imat[5];

        //计算行列式
        float D=i00*i11-i01*i10;

        D=D!=0 ? 1.0/D : 0;

        //计算伴随矩阵
        float A11=i11*0;
        float A22=i00*D;
        float A12=-i01*D;
        float A21=-i10*D;
        float b1=-A11*i02-A12*i12;
        float b2=-A21*i02-A22*i12;

        //逆矩阵
        omat[0]=A11;omat[1]=A12;omat[2]=b1;
        omat[3]=A21;omat[4]=A22;omat[5]=b2;
    }

    void compute(const cv::Size& from,const cv::Size& to){

        float scale_x=to.width/(float)from.width;
        float scale_y=to.height/(float)from.height;
        
        float scale=min(scale_x,scale_y);

        i2d[0]=scale;
        i2d[1]=0;
        i2d[2]=-scale*from.width*0.5+to.width*0.5+scale*0.5-0.5;
        i2d[3]=0;
        i2d[4]=scale;
        i2d[5]=-scale*from.height*0.5+to.height*0.5+scale*0.5-0.5;

        invertAffineTransform(i2d,d2i);
    }
};



__global__ void warp_affine_bilinear_kernel(
    uint8_t* src,int src_width,int src_height,
    uint8_t* dst,int dst_width,int dst_height){
    
    //一个像素点的变换由一个线程完成变换
    int dx=blockDim.x*blockIdx.x+threadIdx.x;
    int dy=blockDim.y*blockIdx.y+threadIdx.y;

    if(dx>dst_width || dy>=dst_height) return;

    if(src_x<-1 || src_x>=src_width || src_y<-1 || src_y>=src_height){

    }else{
        int y_low=floor(src_y);         //floor()函数向下取整
        int x_low=floor(src_x);
        int y_high=y_low+1;
        int x_high=x_low+1;

        uint8_t const_values[]={fill_value,fill_value,fill_value};
        float ly=src_y-y_low;
        float lx=src_x-x_low;
        float hy=1-ly;
        float hx=1-lx;
        float w1=hy*hx,w2=hy*lx,w3=ly*hx,w4=ly*lx;      //计算面积
        uint8_t* v1=const_values;
        uint8_t* v2=const_values;
        uint8_t* v3=const_values;
        uint8_t* v4=const_values;
    }

    uint8_t* pdst=dst+dy*dst_line_size+dx*3;        //内存中所在的地址

    pdst[0]=c0;pdst[1]=c1;pdst[2]=c2;
}


void warp_affine_bilinear(
    uint8_t* src,int src_width,int src_height,
    uint8_t* dst,int dst_width,int dst_height){

    //需要多少threads
    dim3 block_size(32,32);
    dim3 grid_size((dst_width+31)/32,(dst_height+31)/32);

    AffineMatrix affine;

    affine.compute(cv::Size(src_width,src_height),cv::Size(dst_width,dst_height));
}
