# RetinexRawMamba
Paper is available at [arXiv](https://arxiv.org/pdf/2409.07040)

Clone this repository
```
git clone https://github.com/Cynicarlos/RetinexRawMamba.git
cd RetinexRawMamba
```
## Environments Preparation
My cuda version: 11.7  
```
conda create -n RetinexRawMamba python=3.9
conda activate RetinexRawMamba
```
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Download these two files from the following links and upload them to the server, and then install:  
Note: find the corresponding veresion
1. [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.0.0)
`causal_conv1d-1.0.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`
2. [mamba_ssm](https://github.com/state-spaces/mamba/releases/tag/v1.0.1)
`mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`  
you can also download them easily [here](https://drive.google.com/drive/folders/1lsb6MfmGF8OmhqaishnBc69TFNxsabHP)

```
pip install causal_conv1d-1.0.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl  
pip install mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

```
pip install -r requirements.txt
```
## Dataset Preparation
| Dataset | Download link |  Source  |  CFA     |
| :---:   |    :----:     |  :---:   |  :---:   |
| Sony    | [Google Drive](https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  Bayer  |
| Fuji    | [Google Drive](https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  X-Trans  |
| MCR     | [Google Drive](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view)       | [Link](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark)   |  Bayer  |

Note that for SID Sony dataset, to be consistent with DNF, please use the ```Sony_test_list.txt``` we provide in the ```datasets``` folder to evaluate, and there are totally ```562``` images to be tested.  
The directory for the datasets should be as following:  

```
📁datasets/  
├─── 📁MCR/  
│    ├─── 📄MCR_test_list.txt  
│    ├─── 📄MCR_train_list.txt  
│    └─── 📁Mono_Colored_RAW_Paired_DATASET/  
│         ├─── 📁Color_RAW_Input/
│         │    ├─── 📄C00001_48mp_0x8_0x00ff.tif
│         │    └─── 📄...
│         └─── 📁RGB_GT/
│              ├─── 📄C00001_48mp_0x8_0x2fff.jpg
│              └─── 📄...
└─── 📁SID/  
     ├─── 📁Fuji/  
     │    ├─── 📄Fuji_test_list.txt  
     │    ├─── 📄Fuji_train_list.txt  
     │    ├─── 📄Fuji_val_list.txt  
     │    └─── 📁Fuji/  
     │         ├─── 📁Long/
     │         │    ├─── 📄00001_00_10s.RAF
     │         │    └─── 📄...
     │         └─── 📁Short/
     │              ├─── 📄00001_00_0.1s.RAF
     │              └─── 📄...
     └─── 📁Sony/  
          ├─── 📄Sony_test_list.txt  
          ├─── 📄Sony_train_list.txt  
          ├─── 📄Sony_val_list.txt  
          └─── 📁Sony/  
               ├─── 📁Long/
               │    ├─── 📄00001_00_10s.ARW
               │    └─── 📄...
               └─── 📁Short/
                    ├─── 📄00001_00_0.1s.ARW
                    └─── 📄...
```
## Train from scratch
```python
python train.py -cfg configs/sony.yaml
```  
If you want to train on other dataset, just make sure you have the correct config file in the ```configs``` folder, and change the `-cfg` to your own config path.

## Evaluate
Before evaluating our pretrained models, please download them by the following links and put them in the ```pretrained``` folder.  
| Dataset | Pretrained Model  |
| :---:   |    :----:     | 
| Sony    | [Google Drive](https://drive.google.com/file/d/1eAgm5HHDH0CBUsl-czZ7Kdues3tAPy7W/view?usp=drive_link)    [pan baidu](https://pan.baidu.com/s/1G7ytzI0Wd-FLS63UlZI-wA?pwd=x4i4)|
| Fuji    | [Google Drive](https://drive.google.com/file/d/1C9x-VcHdkFt-7MQONSkZAWtttu3Gtp12/view?usp=drive_link)    [pan baidu](https://pan.baidu.com/s/1JNZkoUkBwn_7s0KtkJxo4Q?pwd=w38i)|
| MCR     | [Google Drive](https://drive.google.com/file/d/1OOuyC7PcODPrcNm1uXx2CZwIS8mchtj7/view?usp=drive_link)    [pan baidu](https://pan.baidu.com/s/1_GjDOkKOLPDIASQveiKRUg?pwd=u3an)|

For MCR dataset: 
```python
python test_mcr.py
```  
For SID dataset:  
If your GPU memory is smaller than 40G, generally 24G, please set ```merge_test: true``` in the corresponding config file so that you can test without OOM(out of memory). Otherwise, please set ```merge_test: false``` to get the same results reported in our paper. Note that the results may be a little bit smaller when merge testing than with whole image.
```python
python test_sony.py
```  
 
## Citation
If there is any help for your research, please star this repository and if you want to follow this work, you can cite as following:
```md
@misc{chen2024retinexrawmambabridgingdemosaicingdenoising,
      title={Retinex-RAWMamba: Bridging Demosaicing and Denoising for Low-Light RAW Image Enhancement}, 
      author={Xianmin Chen and Peiliang Huang and Xiaoxu Feng and Dingwen Zhang and Longfei Han and Junwei Han},
      year={2024},
      eprint={2409.07040},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.07040}, 
}
```

## Acknowledgement
The repository is refactored based on [DNF](https://github.com/Srameo/DNF), thanks to the author.
