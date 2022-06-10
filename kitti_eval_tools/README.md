# kitti-object-eval-python

Fast kitti object detection eval in python(finish eval in less than 10 second), support 2d/bev/3d/aos.  

The kitti val result in predict path format: ***.bin***

## Dependencies

Only support python 3.6+, need `numpy`, `skimage`, `numba`, `processbar`, `argparse`

Need `eval_gt_annos.pkl`, `eval_infos.pkl`, `kitti_object_eval_python`

## Usage
```
python eval_kitti.py --pre_path output
```
