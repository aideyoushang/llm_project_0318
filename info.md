## 这里有服务器所有的信息
Linux vm-22a-2fg76 5.4.0-216-generic #236-Ubuntu SMP Fri Apr 11 19:53:21 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

### (llm_csw) root@vm-22a-2fg76:~/csw_test# cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.1 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.1 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal

### (llm_csw) root@vm-22a-2fg76:~/csw_test# python --version
Python 3.10.19
(llm_csw) root@vm-22a-2fg76:~/csw_test# which python
/miniconda3/envs/llm/bin/python

### (llm_csw) root@vm-22a-2fg76:~/csw_test# conda --version
conda 26.1.1
(llm_csw) root@vm-22a-2fg76:~/csw_test# conda env list
base                     /miniconda3
llm                  *   /miniconda3/envs/llm

(llm_csw) root@vm-22a-2fg76:~/csw_test# which conda
/miniconda3/condabin/conda

### (llm_csw) root@vm-22a-2fg76:~/csw_test# nvidia-smi
Wed Mar 18 21:14:29 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:00:06.0 Off |                  N/A |
| 30%   27C    P8             32W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1153      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
print(torch.version.cuda)
12.1
torch.version = 2.4.1




