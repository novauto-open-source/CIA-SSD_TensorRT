/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 16:48:34
 * @LastEditTime: 2019-08-22 17:06:20
 * @LastEditors: Please set LastEditors
 */
#ifndef _ENTROY_CALIBRATOR_H
#define _ENTROY_CALIBRATOR_H

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType,
                int batchSize,const std::vector<std::string>& dataPath,
                const std::string& calibrateCachePath, int bindingSize);

class Int8EntropyCalibrator2 : public nvinfer1::IInt8MinMaxCalibrator {
public:
    Int8EntropyCalibrator2(const int batchSize, const std::vector<std::string>& dataPath,
                           const std::string& calibrateCachePath, int bindingSize);

    virtual ~Int8EntropyCalibrator2();

    int getBatchSize() const noexcept override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int mBatchSize;
    std::vector<std::string> mFileList;
    std::string mCalibrateCachePath;
    int mCurBatchIdx=0;
    int mCount;
    std::vector<void*> mDeviceBatchData;
    std::vector<char> mCalibrationCache;
    std::vector<int> bingsizes;
};


#endif //_ENTROY_CALIBRATOR_H
