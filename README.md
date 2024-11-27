# SR-IQA (Slightly modified from [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch))

This evaluation code aims to facilitate fair comparisons between (real-world) image super-resolution papers.

### Path
* Modify the `path/to/results` at Line 40 and Line 43 in `inference.py` to the path of your SR-IQA folder.

### Installation
```
git clone https://github.com/liyuantsao/SR-IQA.git
cd SR-IQA
pip install -r requirements.txt
python setup.py develop
```

### Basic Usage 
```
# list all available metrics
pyiqa -ls

# test with arbitrary number of metrics
python inference.py --input [Your output image_path or dir] --ref [Ground truth image_path or dir] --save_file results/<EXP_NAME> -m psnr ssim lpips musiq maniqa clipiqa
```
