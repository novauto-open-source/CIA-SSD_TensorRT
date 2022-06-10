/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2020-03-02 15:16:08
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <libgen.h>
#include <math.h>
#include <dirent.h>
#include <cuda_fp16.h>
#include "utils.h"

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

cv::Mat MEAN, STD, IMAGESIZE;
int MODE, USE_MESN, CUDA_INDEX, SAVE_IMAGE, FUSIONMODEL;
std::string ENGINE, OUTNODE;

void SplitString(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void GetFileNames(std::string path, std::string rege, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            std::string tempname = ptr->d_name;
            if (strstr(tempname.c_str(), rege.c_str()) == nullptr){
                continue;
            }
            filenames.push_back(path + "/" + tempname);
        }
    }
    closedir(pDir);
}

void loadCalib(const std::string& filename, float* lidar2rect, float* rect2img){
    ifstream myfile(filename);
	string per_l;
	vector<string> P2;
	vector<string> R0;
	vector<string> V2C;
	while(getline(myfile,per_l)){
		if (per_l.compare(0, 2, "P2") == 0){
			SplitString(per_l, P2, " ");
		}
		else if (per_l.compare(0, 7, "R0_rect") == 0){
			SplitString(per_l, R0, " ");
		}if (per_l.compare(0, 14, "Tr_velo_to_cam") == 0){
			SplitString(per_l, V2C, " ");
		}
	}

    float* cam2img = (float*)malloc(sizeof(float)*16);
	float* r0 = (float*)malloc(sizeof(float)*9);
	float* v2c = (float*)malloc(sizeof(float)*12);
	float* lidar2cam = (float*)malloc(sizeof(float)*16);

	for (int i = 1; i < 13; i++){
		cam2img[i-1] = stof(P2[i]);
		if (i < 10){
			r0[i-1] = stof(R0[i]);
		}
		v2c[i-1] = stof(V2C[i]);
	}

	cam2img[12] = 0.0;
	cam2img[13] = 0.0;
	cam2img[14] = 0.0;
	cam2img[15] = 1.0;

    for (int i = 0; i < 4; i++){
        rect2img[i+4*0] = cam2img[i*4+0];
        rect2img[i+4*1] = cam2img[i*4+1];
        rect2img[i+4*2] = cam2img[i*4+2];
        rect2img[i+4*3] = cam2img[i*4+3];
    }

	cv::Mat r0_m (3, 3, CV_32F, r0);
	cv::Mat v2c_m (3, 4, CV_32F, v2c);

    cv::Mat lidar2cam_m = r0_m*v2c_m;
	memcpy(lidar2cam, lidar2cam_m.ptr<float>(0), sizeof(float)*12);
	lidar2cam[12] = 0.0;
	lidar2cam[13] = 0.0;
	lidar2cam[14] = 0.0;
	lidar2cam[15] = 1.0;

    for (int i = 0; i < 4; i++){
        lidar2rect[i+4*0] = lidar2cam[i*4+0];
        lidar2rect[i+4*1] = lidar2cam[i*4+1];
        lidar2rect[i+4*2] = lidar2cam[i*4+2];
        lidar2rect[i+4*3] = lidar2cam[i*4+3];
    }

    free(cam2img);
    free(r0);
    free(v2c);
    free(lidar2cam);
}

template<typename T>
int loadBinmyself(const std::string& fileName, T* outBuffer, int bytes_size)
{
    std::cout<<std::endl<<"fileName:"<<fileName<<std::endl;
    std::fstream infile(fileName.c_str(), std::ios::in | std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (bytes_size < length){
        length = bytes_size;
    }
    printf(" in loadBinmyself num: %d\n", length/16);

    int stride = 16; 
    for (int i = 0; i < length; i += stride) 
        infile.read((char *)(outBuffer) + i, stride);

    
    infile.close();
    return length / 16;
}


void loadImage(const std::string& img_path, float* buffer, int* shape){
    std::cout << img_path << std::endl;
    cv::Mat img;
    cv::Mat image = cv::imread(img_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int h = image.rows;
    int w = image.cols;
    int c = image.channels();

    shape[0] = h;
    shape[1] = w;

    int downFill = IMAGESIZE.at<int>(0,1) - h;
    int rightFill = IMAGESIZE.at<int>(0,0) - w;

    // image.convertTo(img, CV_32FC3, 1.0/255.0);

    cv::copyMakeBorder(image, image, 0, downFill, 0, rightFill, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    image.convertTo(img, CV_32FC3, 1.0);

    // buffer = img.ptr<float>(0);


    for (int z = 0;z < 3;z++){
        for (int i = 0; i < IMAGESIZE.at<int>(0,1);i++){
            for (int j = 0; j < IMAGESIZE.at<int>(0,0);j++){
                if (USE_MESN){
                    buffer[i*IMAGESIZE.at<int>(0,0)*3+j*3+z] = (img.at<cv::Vec3f>(i,j)[z] - MEAN.at<float>(0,z)) / STD.at<float>(0,z);
                }else{
                    buffer[i*IMAGESIZE.at<int>(0,0)*3+j*3+z] = img.at<cv::Vec3f>(i,j)[z];
                }
            }
        }
    }
    image.release();
    img.release();
}



std::vector<cv::String> binnames;


void test_onnx(const std::string& onnxModelpath, const std::vector<std::string> &dataFile, const std::string& cali_path) {
    std::string engineFile = ENGINE;
    // const std::vector<std::string> customOutput;
    const std::vector<std::string> customOutput {OUTNODE};
    int maxBatchSize = 1;
    int mode = MODE;

    std::vector<std::string> cali_files;
    if (!cali_path.empty()){
        GetFileNames(cali_path, "bin", cali_files);
    }

    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, cali_files);

    std::vector<cv::String> pointcloud;

    cv::glob(dataFile[0],pointcloud,true);//file dir

    size_t firstSize = onnx_net->GetBindingSize(0);
    size_t secondSize = onnx_net->GetBindingSize(1);

    nvinfer1::Dims first_in_dim = onnx_net->GetBindingDims(0);
    int first_shape_size = 1;
    for (int i = 0; i < first_in_dim.nbDims;i++){
        first_shape_size*=first_in_dim.d[i];
    }

    nvinfer1::Dims second_in_dim = onnx_net->GetBindingDims(1);
    int second_shape_size = 1;
    for (int i = 0; i < second_in_dim.nbDims;i++){
        second_shape_size*=second_in_dim.d[i];
    }

    void* indata = malloc(first_shape_size*sizeof(float));
    void* in1 = malloc(second_shape_size*sizeof(int));

    std::vector<void *> inputs;
    inputs.push_back(indata);
    inputs.push_back(in1);
    int points_num;
    int byteSize = first_shape_size*sizeof(float);

    binnames=pointcloud;

    double allpretime = 0.0;
    for(int num=0;num<pointcloud.size();num++)
    {
        auto t1=std::chrono::steady_clock::now();
        points_num = loadBinmyself<float>(pointcloud[num], reinterpret_cast<float*>(inputs[0]), byteSize);   //do inference
        (reinterpret_cast<int*>(inputs[1]))[0] = points_num;
        auto t10=std::chrono::steady_clock::now();
        double per_time=std::chrono::duration<double,std::milli>(t10-t1).count();
        allpretime+=per_time;

        onnx_net->Forward_mult_FP32(inputs, SAVE_IMAGE);
        printf(" \n inference is finished \n");
    }
    printf("\033[0;1;33;41m infer average time is %f, datapre time is %f, copy time is %f\033[0m\n",totaltime/pointcloud.size(), allpretime/pointcloud.size(), copp/pointcloud.size());   
    free(indata);
    free(in1);  
    delete onnx_net;
}

void test_onnx_epnet(const std::string& onnxModelpath, const std::vector<std::string> &dataFile) {
    std::string engineFile = ENGINE;
    // const std::vector<std::string> customOutput;
    const std::vector<std::string> customOutput {OUTNODE};
    std::vector<std::string> cali_files;
    int maxBatchSize = 1;
    int mode = MODE;

    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, cali_files);

    std::vector<cv::String> pointcloud;
    std::string points_path = dataFile[0]+"/velodyne_reduced/";
    cv::glob(points_path,pointcloud,true);
    binnames = pointcloud;

    nvinfer1::Dims points_dim = onnx_net->GetBindingDims(0);
    nvinfer1::Dims valid_dim = onnx_net->GetBindingDims(1);
    nvinfer1::Dims image_dim = onnx_net->GetBindingDims(2);
    nvinfer1::Dims lidar2rect_dim = onnx_net->GetBindingDims(3);
    nvinfer1::Dims rect2img_dim = onnx_net->GetBindingDims(4);
    nvinfer1::Dims imshape_dim = onnx_net->GetBindingDims(5);

    float* points = (float*)malloc(sizeof(float)*volume(points_dim));
    int* valid = (int*)malloc(sizeof(int)*volume(valid_dim));
    float* image = (float*)malloc(sizeof(float)*volume(image_dim));
    // float* image = (float*)malloc(sizeof(float)*1474560);
    float* lidar2rect = (float*)malloc(sizeof(float)*volume(lidar2rect_dim));
    float* rect2img = (float*)malloc(sizeof(float)*volume(rect2img_dim));
    int* imgshape = (int*)malloc(sizeof(int)*volume(imshape_dim));


    for (int i = 0; i < pointcloud.size(); i++){

        std::string points_path = pointcloud[i];
        std::string prue_name = points_path.substr(points_path.find_last_of("/"));
        std::string img_path = dataFile[0]+"/image_2/"+prue_name.substr(1, prue_name.size()-5) + ".png";
        std::string cali_path = dataFile[0]+"/calib/"+prue_name.substr(1, prue_name.size()-5) + ".txt";;

        memset(points, 0.0, sizeof(float)*volume(points_dim));
        int point_num = loadBinmyself<float>(points_path, points, sizeof(float)*volume(points_dim));
        valid[0] = point_num;
        loadImage(img_path, image, imgshape);
        loadCalib(cali_path, lidar2rect, rect2img);

        // std::vector<void*> input_datas{points, valid, trans, trans+volume(lidar2rect_dim), imgshape};  //, trans+volume(lidar2rect_dim)
        std::vector<void*> input_datas{points, valid, image, lidar2rect, rect2img, imgshape};  //, trans+volume(lidar2rect_dim)
        onnx_net->Forward_mult_FP32(input_datas, SAVE_IMAGE);
    }
    printf("\033[0;1;33;41m average time %f\033[0m\n",totaltime/pointcloud.size());

    free(points);
    free(valid);
    free(image);
    free(lidar2rect);
    free(rect2img);
}


class InputParser{                                                              
    public:                                                                     
        InputParser (int &argc, char **argv){                                   
            for (int i=1; i < argc; ++i)                                        
                this->tokens.push_back(std::string(argv[i]));                   
        }                                                                       
        const std::string& getCmdOption(const std::string &option) const{       
            std::vector<std::string>::const_iterator itr;                       
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option); 
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){      
                return *itr;                                                    
            }                                                                   
            static const std::string empty_string("");                          
            return empty_string;                                                
        }                                                                       
        bool cmdOptionExists(const std::string &option) const{                  
            return std::find(this->tokens.begin(), this->tokens.end(), option)  
                   != this->tokens.end();                                       
        }                                                                       
    private:                                                                    
        std::vector <std::string> tokens;                                       
};  

int main(int argc, char** argv) {
    cv::FileStorage fs("../config.xml", cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["MEAN"] >> MEAN;
        fs["STD"] >> STD;
        fs["IMAGESIZE"] >> IMAGESIZE;
        fs["MODE"] >> MODE;
        fs["USE_MESN"] >> USE_MESN;
        fs["CUDA_INDEX"] >> CUDA_INDEX;
        fs["SAVE_IMAGE"] >> SAVE_IMAGE;
        fs["ENGINE"] >> ENGINE;
        fs["OUTNODE"] >> OUTNODE;
        fs["FUSIONMODEL"] >> FUSIONMODEL;
        fs.release();
    }
    cudaSetDevice(CUDA_INDEX);
    InputParser cmdparams(argc, argv);

    const std::string& trt_path = cmdparams.getCmdOption("--onnx_path");
    const std::string& data_path = cmdparams.getCmdOption("--data_path");
    const std::string& cali_path = cmdparams.getCmdOption("--cali_path");
    std::vector<std::string> allIn{data_path};
    if (FUSIONMODEL){
        test_onnx_epnet(trt_path, allIn);
    }else{
        test_onnx(trt_path, allIn, cali_path);
    }
    
    return 0;
}
