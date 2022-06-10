#include "Int8Calibrator.h"
#include "spdlog/spdlog.h"
#include "utils.h"

#include <fstream>
#include <iterator>
#include <cassert>
#include <string.h>
#include <algorithm>

#include <sys/types.h>
#include <dirent.h>

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        if(strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0 ) {
            continue;
        }
        v.push_back(name+dp->d_name);
    }
    closedir(dirp);
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

    int stride = 16; 
    for (int i = 0; i < length; i += stride) 
        infile.read((char *)(outBuffer) + i, stride);

    
    infile.close();
    return length / 16;
}

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType,
                int batchSize,const std::vector<std::string>& dataPath,
                const std::string& calibrateCachePath, int bindingSize) {
    if(calibratorType == "Int8EntropyCalibrator2") {
        return new Int8EntropyCalibrator2(batchSize,dataPath,calibrateCachePath, bindingSize);
    } else {
        spdlog::error("unsupported calibrator type");
        assert(false);
    }
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(const int batchSize, const std::vector<std::string>& dataPath,
                                               const std::string& calibrateCachePath, int bindingSize)
{
    spdlog::info("init calibrator...");
    mBatchSize = batchSize;
    mCalibrateCachePath = calibrateCachePath;

    if(dataPath.size()>0) {
        mFileList = dataPath;
        mCount = mFileList.size();
        assert(mCount != 0);

        mDeviceBatchData.resize(2);

        mDeviceBatchData[0] = safeCudaMalloc(bindingSize * mBatchSize);
        mDeviceBatchData[1] = safeCudaMalloc(4 * mBatchSize);
        bingsizes.push_back(bindingSize);
        bingsizes.push_back(4);
    }
}


Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    for(size_t i=0;i<mDeviceBatchData.size();i++) {
        safeCudaFree(mDeviceBatchData[i]);
    }
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept{
    spdlog::info("get batch size {}", mBatchSize);
    return mBatchSize;
}


bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (mCurBatchIdx + mBatchSize > mCount) {
        return false;
    }

    float* first_data = (float*)malloc(bingsizes[0]);
    int* second_data = (int*)malloc(bingsizes[1]);
    for(int i=0;i<mBatchSize; i++) {
        memset(first_data, 0.0, bingsizes[0]);
        int length = loadBinmyself<float>(mFileList[mCurBatchIdx], first_data, bingsizes[0]);
        second_data[0] = length;
        for(int j=0;j<nbBindings;j++) {
            void* p = static_cast<char*>(mDeviceBatchData[j]) + i*bingsizes[j];
            if (j == 0){
                CUDA_CHECK(cudaMemcpy(p, first_data, bingsizes[j], cudaMemcpyHostToDevice));
            }else{
                CUDA_CHECK(cudaMemcpy(p, second_data, bingsizes[j], cudaMemcpyHostToDevice));
            }
        }
        mCurBatchIdx++;
    }
    free(first_data);
    free(second_data);

    for(int j=0;j<nbBindings;j++) {
        bindings[j] = mDeviceBatchData[j];
    }
    spdlog::info("load catlibrate data {}/{} done", mCurBatchIdx, mCount);
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    spdlog::info("read calibration cache");
    mCalibrationCache.clear();
    std::ifstream input(mCalibrateCachePath, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
        std::copy(std::istream_iterator<char>(input),
                  std::istream_iterator<char>(),
                  std::back_inserter(mCalibrationCache));
    }

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    spdlog::info("write calibration cache");
    std::ofstream output(mCalibrateCachePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}