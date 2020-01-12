# Neural-Module-Networks.Tensorlayer

A tensorlayer implementation of Neural Module Networks.


## Learning to Reason: End-to-End Module Networks for Visual Question Answering

This project refered to" [Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://github.com/ronghanghu/n2nmn)" and implemented the model with tensorlayer. It contains the code for the following paper (with tests on the SHAPES dataset):

* R. Hu, J. Andreas, M. Rohrbach, T. Darrell, K. Saenko, *Learning to Reason: End-to-End Module Networks for Visual Question Answering*. in ICCV, 2017. ([PDF](https://arxiv.org/pdf/1704.05526.pdf))
```
@inproceedings{hu2017learning,
  title={Learning to Reason: End-to-End Module Networks for Visual Question Answering},
  author={Hu, Ronghang and Andreas, Jacob and Rohrbach, Marcus and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```


## Installation

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install TensorFlow v1.0.0 (Note: newer or older versions of TensorFlow may fail to work due to incompatibility with TensorFlow Fold):  
`pip install tensorflow-gpu==1.0.0`  
3. Install [TensorFlow Fold](https://github.com/tensorflow/fold) (which is needed to run dynamic graph):  
`pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl`
4. Install tensorlayer1.4.1: 
`pip install tensorlayer==1.4.1`
5. Install cuda & cudnn (in Anaconda): 
`connda install cuda`
`connda install cudnn==5.1`
5. Download this or clone with Git, and then enter the root directory of the repository:  
`git clone https://github.com/jiaqi-xi/Neural-Module-Networks.Tensorlayer.git`

## Train and evaluate on the SHAPES dataset

A copy of the SHAPES dataset is contained in this repository under `exp_shapes/shapes_dataset`. The ground-truth module layouts (expert layouts) we use in our experiments are also provided under `exp_shapes/data/*_symbols.json`. The script to obtain the expert layouts from the annotations is in `exp_shapes/data/get_ground_truth_layout.ipynb`.

### Training

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Train with ground-truth layout (behavioral cloning from expert):  
`python exp_shapes/train_shapes_gt_layout.py`  

2. Train without ground-truth layout (policy search from scratch):  
`python exp_shapes/train_shapes_scratch.py`  

Note: by default, the above scripts use GPU 0. To train on a different GPU, set the `--gpu_id` flag. During training, the script will write TensorBoard events to `exp_shapes/tb/` and save the snapshots under `exp_shapes/tfmodel/`.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate *shapes_gt_layout* (behavioral cloning from expert):  
`python exp_shapes/eval_shapes.py --exp_name shapes_gt_layout --snapshot_name 00040000 --test_split test`  

2. Evaluate *shapes_scratch* (policy search from scratch):  
`python exp_shapes/eval_shapes.py --exp_name shapes_scratch --snapshot_name 00400000 --test_split test`  

Note: the above evaluation scripts will print out the accuracy and also save it under `exp_shapes/results/`. By default, the above scripts use GPU 0, and evaluate on the *test* split of SHAPES. To evaluate on a different GPU, set the `--gpu_id` flag. To evaluate on the *validation* split, use `--test_split val` instead.
