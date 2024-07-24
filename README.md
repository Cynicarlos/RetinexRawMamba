# TSCDNet (Tow Stage Cross Domain Net)
```
git clone https://github.com/Cynicarlos/TSCDNet.git
cd TSCDNet
```
## Train from scratch
Envs of Mamba: My cuda version:11.7  
```
conda create -n TSCDNet python=3.9
conda activate TSCDNet
```
Download these two files from the following links and upload them to the server, and then install:  
Note: find the corresponding veresion
1. [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.0.0)
`causal_conv1d-1.0.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`
2. [mamba_ssm](https://github.com/state-spaces/mamba/releases/tag/v1.0.1)
`mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install causal_conv1d-1.0.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl  
pip install mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install requirements.txt
```

One example:  
```python
python train.py --cfg configs/sony.yaml
```  
If you want to train on other dataset, just make sure you have the correct config file and change the `--cfg` to your own config path.

## Evaluate
### Dataset Preparation
You can access [SID_Sony](https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view) and [SID_Fuji](https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view), more details [here](https://github.com/cchen156/Learning-to-See-in-the-Dark)  
[MCR](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view), more details [here](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark).  

Before evaluating our pretrained models, please download them [sony_best_model.pth](https://drive.google.com/file/d/1eAgm5HHDH0CBUsl-czZ7Kdues3tAPy7W/view?usp=drive_link), [fuji_best_model.pth](https://drive.google.com/file/d/1C9x-VcHdkFt-7MQONSkZAWtttu3Gtp12/view?usp=drive_link), [mac_best_model.pth](https://drive.google.com/file/d/1OOuyC7PcODPrcNm1uXx2CZwIS8mchtj7/view?usp=drive_link), and put them in the ```pretrained``` folder.  

For MCR dataset: 
```python
python test_MCR.py
```  
For SID dataset:  
If your GPU memory is larger than 40G, just 
```python
python test_SID_Sony.py
```
or
```python
python test_SID_Fuji.py
```
Otherwise, please set 
```
merge_test: true
```
in the corresponding config file, note that the results may be a little bit smaller than tesing with whole image.
