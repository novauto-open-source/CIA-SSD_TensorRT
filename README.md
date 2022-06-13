# A TRT Inference Tool

This repository is a tool for TensorRT inference. It is suited for xavier agx(jetpack4.6). We provide two onnx models(CIASSD, float and integer) and kitti validation set. This repository also provides a precision eval tools based on python.


## Set LD_LIBRARY_PATH

```bash
$ export LD_LIBRARY_PATH=$(pwd)/nova3d_lib/:$LD_LIBRARY_PATH
```

## Compile

```bash
$ mkdir build
$ cmake ..
$ make -j8
```


## Test

```bash
$ # run fp32 or fp16    edit config.xml, MODE 0 or 1 
$ ./unit_test --onnx_path ../model/ciassd_float.onnx.onnx --data_path ../validation_set/
$ # run int8    edit config.xml, MODE 2
$ ./unit_test --onnx_path ../model/ciassd_quant.onnx.onnx --data_path ../validation_set/
```
Result will be saved in "NovaTrt/output/".


NOTE:

For the Kitti dataset, use the reduced bins to replicate our results.

Here are parameters in [config.xml](config.xml):

```model(test.xml)
<?xml version="1.0"?>
<opencv_storage>
<MEAN type_id="opencv-matrix">   # image settings
  <rows>3</rows>
  <cols>1</cols>
  <dt>f</dt>
  <data>
    4.85000014e-01 4.56000000e-01 4.05999988e-01</data></MEAN>
<STD type_id="opencv-matrix">   # image settings
  <rows>3</rows>
  <cols>1</cols>
  <dt>f</dt>
  <data>
    2.29000002e-01 2.24000007e-01 2.24999994e-01</data></STD>
<IMAGESIZE type_id="opencv-matrix">   # image settings
  <rows>2</rows>
  <cols>1</cols>
  <dt>i</dt>
  <data>
    1280 384</data></IMAGESIZE>
<USE_MESN>0</USE_MESN>   # image settings
<MODE>0</MODE>  #  inference mod, 0 for fp32, 1 for fp16, 2 for int8
<SAVE_IMAGE>1</SAVE_IMAGE>  #  save output or not
<CUDA_INDEX>0</CUDA_INDEX>  #  which cuda being used
<ENGINE>""</ENGINE>   #   the location of engine file
<OUTNODE>""</OUTNODE>   #   set the custom output for onnx model
<FUSIONMODEL>0</FUSIONMODEL>  #  if use lidar and image fusion model
<!-- "../engine/pointrcnn.trt" -->
</opencv_storage>
```

## Evaluation

[kitti_eval_tools](kitti_eval_tools) is a python program.

```bash
python3 eval_kitti.py --pre_path ../output/
```
## A reference result:

|  | 3D AP_11(%) | 3D AP_40(%) | inference time(ms) |
| :------: | :------: | :------: |:----:|
| FP32 | 80.0 | 83.8 | 123.2 |
| FP16 | 80.0 | 83.9 | 91.2 |
| INT8 | 78.7 | 81.9 | 65.0 |


## Contributors

Hao Liu, Zhongyuan Qiu, yifei chen, yali zhao. 

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](novauto.png)


## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.