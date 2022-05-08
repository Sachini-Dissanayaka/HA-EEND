<h1 align="center">Attention-based End-to-End Neural Diarization</h1>

This repository comprises source code for two main research objectives
1. Combining various attention mechanisms to obtain a better model for two-speaker overlapping speech speaker diarization than the current state-of-the-art approaches. <br>
The following combined attention mechanisms have been employed in the work. Combined as well as single attention mechanisms can be obtained by commenting the respective lines of code from ``` pytorch_backend/models.py  ```
-  Self Attention + Local Dense Synthesizer Attention (HA-EEND)
-  External Attention + Local Dense Synthesizer Attention
-  Relative Attention + Local Dense Synthesizer Attention
2. Experiments on the language dependency of EEND-based speaker diarization, and testing on combined datasets in both English and Sinhala languages

The repository largely references code from the following sources: 
- [EEND](https://github.com/hitachi-speech/EEND) by [Research & Development Group, Hitachi, Ltd.](https://github.com/hitachi-speech) who holds the copyright
- [EEND_PyTorch](https://github.com/Xflick/EEND_PyTorch) licensed under [MIT License](https://github.com/Xflick/EEND_PyTorch/blob/master/LICENSE)
- [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch) licensed under [MIT License](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/LICENSE)
- [multihead-LDSA](https://github.com/mlxu995/multihead-LDSA)
- [attentions](https://github.com/sooftware/attentions) licensed under [MIT License](https://github.com/sooftware/attentions/blob/master/LICENSE)
- [ASR Recipes](https://github.com/google/asr-recipes) licensed under an [Apache License, Version 2.0.](https://github.com/google/asr-recipes/blob/master/LICENSE)

## Directory Structure

```
├── egs : middle tier files                   
    ├── asr-sinhala/v1 : Modelling on Sinhala ASR and CALLSINHALA
        ├── conf : configuration files
        ├── local : locally used scripts and other files
        ├── cmd.sh : file that specifies job scheduling system
        ├── path.sh : path file
        ├── run.sh : train/infer/score model
        └── run_prepare_shared.sh : prepare data
    ├── callhome/v1 : CALLHOME test set
    ├── combined/v1 : Combined modelling on Sinhala ASR/LibriSpeech and test on CALLHOME
    └── librispeech/v1 : Modelling on LibriSpeech and CALLHOME
├── eend : backend files  
    └── pytorch_backend/models.py : specify different models to be trained on
└── tools : Kaldi setup       
```

## Installing requirements and Setting-up

The research was conducted in the following environment <br>
- OS : Ubuntu 18.04 LTS
- Memory: 
  - For single multi-head layered encoder blocks: 8 CPUs, 32 GB RAM
  - For double multi-head layered encoder blocks: 16 CPUs, 64 GB RAM
- Storage : 150-200 GB

The following requirements are to be installed 
- Anaconda
- CUDA Toolkit
- SoX tool

Follow the following steps to install all the requirements and get going on the project. <br>

#### 1. Install Anaconda

``` 
sudo apt-get update 
sudo apt-get install bzip2 libxml2-dev -y 
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh (use Anaconda latest version)
bash Anaconda3-2020.11-Linux-x86_64.sh
rm Anaconda3-2020.11-Linux-x86_64.sh
source .bashrc 
```

#### 2. Install the required libraries

``` 
sudo apt install nvidia-cuda-toolkit -y
sudo apt-get install unzip gfortran python2.7 -y
sudo apt-get install automake autoconf sox libtool subversion -y
sudo apt-get update -y
sudo apt-get install -y flac
``` 

#### 3. Clone the Git repository

``` 
git clone https://github.com/Sachini-Dissanayaka/HA-EEND.git 
```

#### 4. Install Kaldi and Python environment

``` 
cd HA-EEND/tools/ 
make 
```

#### 5. Install Pytorch

```
~/HA-EEND/tools/miniconda3/envs/eend/bin/pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 6. Add paths

```
export PYTHONPATH="${PYTHONPATH}:~/HA-EEND/"
export PATH=~/HA-EEND/tools/miniconda3/envs/eend/bin/:$PATH
export PATH=~/HA-EEND/eend/bin:~/HA-EEND/utils:$PATH
export KALDI_ROOT=~/HA-EEND/tools/kaldi
export PATH=~/HA-EEND/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:~/HA-EEND:$PATH
```

## Configuration
Modify ```egs/librispeech/v1/cmd.sh``` according to your job schedular.

## Data Preparation

The following datasets were used in the experiments.
- Training
    - [LibriSpeech ASR corpus](https://www.openslr.org/12)
    - [Sinhala ASR corpus](https://openslr.org/52/)
- Testing
    - [CALLHOME portion](https://catalog.ldc.upenn.edu/LDC2001S97) of the 2000 NIST Speaker Recognition Evaluation Corpus
    - CALLSINHALA dataset (collected by the authors)

For tests with English data: <br>
Move the datasets (LibriSpeech and CALLHOME) into a folder with path egs/librispeech/v1/data/local <br>
Run the following commands
```
cd egs/librispeech/v1
./run_prepare_shared.sh
```

## Run training, inference, and scoring
```
./run.sh
```

## Reach us for any further clarifications

 - Yoshani Ranaweera : yoshani.ranaweera.17@cse.mrt.ac.lk
 - Sachini Dissanayaka : sachinidissanayaka.17@cse.mrt.ac.lk
 - Anjalee Sudasinghe : anjaleeps.17@cse.mrt.ac.lk

