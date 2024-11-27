### Modified from [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch)

---
### Installation
```
# Install with git clone
git clone https://github.com/liyuantsao/My-IQA.git
cd My-IQA
pip install -r requirements.txt
python setup.py develop
```

### Basic Usage 
```
# list all available metrics
pyiqa -ls

# test with arbitrary number of metrics
python inference.py --input [your image_path or dir] --ref [ground truth image_path or dir] --save_file results/EXP_NAME -m psnr ssim lpips musiq maniqa clipiqa
```
