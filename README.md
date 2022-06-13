# NOVA-3D

## Introduction

NOVA-3D is the industry's first all-in-one optimization, acceleration and deployment tool that is specifically created for 3D point cloud algorithms. The self-developed solution by NOVAUTO features a complete point cloud operator library, a pruning and quantization tool, a reference model library and an inference acceleration library. NOVA-3D effectively solves the major pain point of 3D point cloud algorithms that are usually too complex for deployment on embedded systems while ensuring high performance and real-time inference of the models. The high level of automation of NOVA-3D greatly reduces project complexity and helps our customers to realize their AI solutions fast, at low costs and with unparalleled performance.

The model library contains numerous state-of-the-art models (voxel-based, point-based, view-based, LiDAR-image, etc.) that can all be deployed in a unified manner on NVIDIA's embedded platforms such as Jetson Xavier or Jetson Orin.

A great example for the effectiveness of NOVA-3D is the included state-of-the-art model CIA-SSD. To give you a first-hand impression, this repository provides everything necessary for the deployment of the NOVA-3D CIA-SSD model on your Jetson Xavier AGX platform (jetpack4.6) with evaluation on the KITTI dataset: We provide two optimized ONNX models (floating point and quantization), the required TensorRT inference tool and the KITTI evaluation code for seamless validation of the results.

With NOVA-3D, our customers can realize their cutting-edge LiDAR perception solutions with accelerated deployment on NVIDIA's embedded platforms such as Jetson Xavier or Jetson Orin. If you are interested, please contact us.

## Getting Started

On your NVIDIA Jetson Xavier AGX (jetpack4.6), compile the program:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
```

Set up your LD_LIBRARY_PATH:

```bash
$ export LD_LIBRARY_PATH=$(pwd)/nova3d_lib/:$LD_LIBRARY_PATH
```

Set the mode in the [config.xml](config.xml) file and run the program:

```bash
# fp32 or fp16:   set MODE=0(fp32) or MODE=1(fp16) 
$ ./unit_test --onnx_path ../model/ciassd_float.onnx --data_path ../validation_set/
# int8:           set MODE=2(int8)
$ ./unit_test --onnx_path ../model/ciassd_quant.onnx --data_path ../validation_set/
```

Evaluate the results (requires python3):

```bash
python3 eval_kitti.py --pre_path ../output/
```


## KITTI Results

*NOVA-3D CIA-SSD on NVIDIA Jetson Xavier AGX:*

|  | 3D AP_11 (%) | 3D AP_40 (%) | inference time (ms) |
| :------: | :------: | :------: |:----:|
| FP32 | 80.0 | 83.8 | 123.2 |
| FP16 | 80.0 | 83.9 | 91.2 |
| INT8 | 79.45 | 83.12 | 65.13 |

To reproduce our results, use the provided KITTI validation set. The point cloud data is reduced to the area visible by the camera.


## Contributors

Hao Liu, Zhongyuan Qiu, Yifei Chen, Yali Zhao

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](novauto.png)


## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

The code is developed based on [TensorRT](https://github.com/NVIDIA/TensorRT).
Thanks for previous works [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD).

## Contact

If you have any question or suggestion about this repo, please contact us(zhongyuan.qiu@novauto.com.cn, hao.liu@novauto.com.cn, yifei.chen@novauto.com.cn).