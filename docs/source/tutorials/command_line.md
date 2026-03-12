---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Using ONTraC via Command Line

This document shows the step-by-step process of running ONTraC using the Stereo-seq mouse midbrain dataset.

## Preparation

### The hardware environment used for this tutorial

#### CPU

```{code-cell}
:class: input-cell

lscpu
```

```{code-block}
:class: output-cell

Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 57 bits virtual
Byte Order:                         Little Endian
CPU(s):                             96
On-line CPU(s) list:                0-95
Vendor ID:                          GenuineIntel
Model name:                         INTEL(R) XEON(R) PLATINUM 8568Y+
CPU family:                         6
Model:                              207
Thread(s) per core:                 1
Core(s) per socket:                 48
Socket(s):                          2
Stepping:                           2
Frequency boost:                    enabled
CPU(s) scaling MHz:                 100%
CPU max MHz:                        2301.0000
CPU min MHz:                        800.0000
BogoMIPS:                           4600.00
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 ds_cpl smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hfi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
L1d cache:                          4.5 MiB (96 instances)
L1i cache:                          3 MiB (96 instances)
L2 cache:                           192 MiB (96 instances)
L3 cache:                           600 MiB (2 instances)
NUMA node(s):                       2
NUMA node0 CPU(s):                  0-47
NUMA node1 CPU(s):                  48-95
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Not affected
Vulnerability Spec rstack overflow: Not affected
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Enhanced / Automatic IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected
```

```{note}
We only use 4 CPUs here.
```

#### GPU

```{code-cell}
:class: input-cell

nvidia-smi
```

```{code-block}
:class: output-cell

Thu Jun  5 15:32:39 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:3D:00.0 Off |                  Off |
| N/A   26C    P0             66W /  700W |       1MiB /  81559MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### ONTraC Installation

```{code-cell}
:class: input-cell

source ~/.bash_profile  # change it to "source ~/.zshrc" if you are using zsh

conda activate ONTraC  # ONTraC is the conda environment name

pip install "ONTraC[analysis]==1.*"
```

:::{dropdown} Show output (Click to expand)
:class: folded-output

```text
Collecting ONTraC==1.* (from ONTraC[analysis]==1.*)
  Downloading ontrac-1.2.0-py3-none-any.whl.metadata (7.0 kB)
Requirement already satisfied: pyyaml==6.0.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (6.0.1)
Requirement already satisfied: pandas==2.2.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (2.2.1)
Requirement already satisfied: scipy<1.15.0a0,>=1.14.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (1.14.1)
Requirement already satisfied: torch<2.7.0a0,>=2.6.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (2.6.0)
Requirement already satisfied: torch-geometric<2.7.0a0,>=2.6.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (2.6.1)
Requirement already satisfied: session-info in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC==1.*->ONTraC[analysis]==1.*) (1.0.0)
Requirement already satisfied: matplotlib<3.9 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC[analysis]==1.*) (3.8.4)
Requirement already satisfied: seaborn in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from ONTraC[analysis]==1.*) (0.13.2)
Requirement already satisfied: numpy<2,>=1.23.2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from pandas==2.2.1->ONTraC==1.*->ONTraC[analysis]==1.*) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from pandas==2.2.1->ONTraC==1.*->ONTraC[analysis]==1.*) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from pandas==2.2.1->ONTraC==1.*->ONTraC[analysis]==1.*) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from pandas==2.2.1->ONTraC==1.*->ONTraC[analysis]==1.*) (2024.1)
Requirement already satisfied: contourpy>=1.0.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (4.51.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (24.0)
Requirement already satisfied: pillow>=8 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (10.3.0)
Requirement already satisfied: pyparsing>=2.3.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from matplotlib<3.9->ONTraC[analysis]==1.*) (3.1.2)
Requirement already satisfied: filelock in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.13.4)
Requirement already satisfied: typing-extensions>=4.10.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (4.11.0)
Requirement already satisfied: networkx in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.3)
Requirement already satisfied: jinja2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.1.3)
Requirement already satisfied: fsspec in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2024.3.1)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.2.0)
Requirement already satisfied: sympy==1.13.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from sympy==1.13.1->torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (1.3.0)
Requirement already satisfied: aiohttp in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.9.5)
Requirement already satisfied: psutil>=5.8.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (5.9.8)
Requirement already satisfied: requests in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2.31.0)
Requirement already satisfied: tqdm in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (4.66.5)
Requirement already satisfied: stdlib-list in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from session-info->ONTraC==1.*->ONTraC[analysis]==1.*) (0.10.0)
Requirement already satisfied: six>=1.5 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas==2.2.1->ONTraC==1.*->ONTraC[analysis]==1.*) (1.16.0)
Requirement already satisfied: aiosignal>=1.1.2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from aiohttp->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from aiohttp->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (23.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from aiohttp->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (1.4.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from aiohttp->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (6.0.5)
Requirement already satisfied: yarl<2.0,>=1.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from aiohttp->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (1.9.4)
Requirement already satisfied: MarkupSafe>=2.0 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from jinja2->torch<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2.1.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from requests->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from requests->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from requests->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in /sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages (from requests->torch-geometric<2.7.0a0,>=2.6.0->ONTraC==1.*->ONTraC[analysis]==1.*) (2024.2.2)
Downloading ontrac-1.2.0-py3-none-any.whl (74 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.8/74.8 kB 7.4 MB/s eta 0:00:00
Installing collected packages: ONTraC
Successfully installed ONTraC-1.2.0
```
:::

### Download Dataset

```{code-cell}
:class: input-cell

curl -O https://zenodo.org/records/15571644/files/Simulated_data.zip
```

```{code-block}
:class: output-cell

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 46.7M  100 46.7M    0     0  18.4M      0  0:00:02  0:00:02 --:--:-- 18.5M
```

```{code-cell}
:class: input-cell

unzip Simulated_data.zip
```

:::{dropdown} Show output (Click to expand)
:class: folded-output

```text
Archive:  Simulated_data.zip
   creating: Simulated_data/
   creating: Simulated_data/ONTraC_output/
   creating: Simulated_data/ONTraC_run_log/
   creating: Simulated_data/preprocessing/
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_16_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_5_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_5_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_base_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_1_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_6_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_100_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_1_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_1_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_100_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_1_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_16_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_5_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_5_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_5_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_16_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_5_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_1_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_100_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_2_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_20_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_base_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_1_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_1_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_base_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_3_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_5_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_2_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_16_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_base_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_base_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_3_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_20_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_2_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_base_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_base_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_3_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_100_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_100_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_1_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_base_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_base_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_16_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_2_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_2_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_20_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_20_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_3_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_base_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_base_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_2_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_16_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_base_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_6_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_2_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_100_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_5_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_2_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_6_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_base_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_6_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_base_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_2_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_5_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_hidden_feats_16_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_1_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_base_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_100_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_1_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_3_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_100_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_1_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_5_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_6_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_4_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_16_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_5_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_16_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_20_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_gcn_layers_4_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_16_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_2_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_20_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_3_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_16_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_6_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_3_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_20_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_8_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_hidden_feats_2_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_1_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_100_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_base_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_base_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_3_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_20_NN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_3_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_6_NT  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_hidden_feats_16_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_3_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_neighbors_20_NN  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_2_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_k_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_8_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_6_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_3_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_20_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_6_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_100_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_k_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_6_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_k_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_10_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_10_GNN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_5_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_k_10_NT  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_20_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_gcn_layers_6_NT  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_hidden_feats_8_GNN  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_disconnected_n_neighbors_100_GNN  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_linear_n_neighbors_100_NN  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_4_GNN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_neighbors_20_NN  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_non_linear_n_gcn_layers_3_NT  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/
  inflating: __MACOSX/Simulated_data/ONTraC_output/._simulated_data_n_gcn_layers_6_NT  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_gcn_layers_1.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_gcn_layers_1.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_neighbors_100.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_neighbors_100.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_hidden_feats_2.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_hidden_feats_2.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_k_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_k_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_neighbors_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_neighbors_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_gcn_layers_3.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_gcn_layers_3.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_base.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_base.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_k_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_k_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_k_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_k_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_gcn_layers_6.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_gcn_layers_6.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_gcn_layers_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_gcn_layers_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_hidden_feats_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_hidden_feats_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_neighbors_100.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_neighbors_100.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_neighbors_20.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_neighbors_20.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_k_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_k_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_gcn_layers_5.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_gcn_layers_5.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_gcn_layers_1.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_gcn_layers_1.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_base.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_base.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_k_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_k_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_hidden_feats_16.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_hidden_feats_16.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_hidden_feats_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_hidden_feats_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_hidden_feats_2.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_hidden_feats_2.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_k_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_k_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_gcn_layers_3.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_gcn_layers_3.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_neighbors_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_neighbors_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_gcn_layers_6.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_gcn_layers_6.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_gcn_layers_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_gcn_layers_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_hidden_feats_16.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_hidden_feats_16.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_neighbors_20.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_neighbors_20.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_gcn_layers_5.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_gcn_layers_5.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_gcn_layers_3.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_gcn_layers_3.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_neighbors_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_neighbors_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_k_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_k_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_neighbors_20.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_neighbors_20.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_hidden_feats_2.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_hidden_feats_2.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_hidden_feats_16.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_hidden_feats_16.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_hidden_feats_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_hidden_feats_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_gcn_layers_1.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_gcn_layers_1.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_gcn_layers_5.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_gcn_layers_5.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_neighbors_100.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_neighbors_100.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_gcn_layers_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_gcn_layers_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_base.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_base.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_gcn_layers_6.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_gcn_layers_6.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_k_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_k_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_gcn_layers_6.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_gcn_layers_6.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_base.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_base.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_disconnected_n_neighbors_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_disconnected_n_neighbors_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_k_10.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_k_10.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_gcn_layers_5.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_gcn_layers_5.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_k_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_k_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_hidden_feats_16.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_hidden_feats_16.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_hidden_feats_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_hidden_feats_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_k_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_k_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_gcn_layers_4.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_gcn_layers_4.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_k_8.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_k_8.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_non_linear_n_neighbors_20.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_non_linear_n_neighbors_20.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_hidden_feats_2.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_hidden_feats_2.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_data_n_neighbors_100.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_data_n_neighbors_100.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_gcn_layers_1.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_gcn_layers_1.log  
  inflating: Simulated_data/ONTraC_run_log/simulated_linear_n_gcn_layers_3.log  
  inflating: __MACOSX/Simulated_data/ONTraC_run_log/._simulated_linear_n_gcn_layers_3.log  
   creating: Simulated_data/preprocessing/Simulated_dataset_1/
   creating: Simulated_data/preprocessing/Simulated_dataset_2/
   creating: Simulated_data/preprocessing/Simulated_dataset_3/
   creating: Simulated_data/preprocessing/Simulated_dataset_4/
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_base_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_base_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_base_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_base_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/Simulation_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._Simulation_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._Simulation_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/Simulation_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._Simulation_CellTypeComposition.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/Simulation_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._Simulation_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/Simulation_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._Simulation_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._processed  
   creating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/simulated_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._simulated_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/simulated_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._simulated_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/simulated_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._simulated_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/simulated_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._simulated_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/cell_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._cell_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/consolidate_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._consolidate_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/niche_level_max_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._niche_level_max_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/Simulation_out.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._Simulation_out.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/Simulation_z.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._Simulation_z.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/epoch_0.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._epoch_0.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/niche_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._niche_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/consolidate_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._consolidate_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/consolidate_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._consolidate_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/Simulation_out_adj.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._Simulation_out_adj.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/Simulation_s.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._Simulation_s.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/model_state_dict.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._model_state_dict.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/cell_level_niche_cluster.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_GNN/._cell_level_niche_cluster.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/simulated_NeighborIndicesMatrix.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._simulated_NeighborIndicesMatrix.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/samples.yaml  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._samples.yaml  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/meta_data.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._meta_data.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/simulated_NicheWeightMatrix.npz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._simulated_NicheWeightMatrix.npz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/simulated_EdgeIndex.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._simulated_EdgeIndex.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/cell_type_code.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._cell_type_code.csv  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/processed/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._processed  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/simulated_Coordinates.csv  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._simulated_Coordinates.csv  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/simulated_CellTypeComposition.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._simulated_CellTypeComposition.csv.gz  
   creating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/raw/
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/._raw  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/simulated_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/._simulated_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NT/._NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/cell_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/._cell_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/Simulation_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/._Simulation_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/niche_cluster_score.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/._niche_cluster_score.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/niche_NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/._niche_NTScore.csv.gz  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/NTScore.csv.gz  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NT/._NTScore.csv.gz  
  inflating: Simulated_data/preprocessing/Simulated_dataset_1/metrics.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_1/._metrics.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_1/full_simulation_data_with_noise.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_1/._full_simulation_data_with_noise.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_1/simulation_data_with_noise.rds  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_1/._simulation_data_with_noise.rds  
  inflating: Simulated_data/preprocessing/Simulated_dataset_1/simulated_input.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_1/._simulated_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_1/simulated_expression.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_1/._simulated_expression.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_2/simulation_data_branch_linear_with_noise.rds  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_2/._simulation_data_branch_linear_with_noise.rds  
  inflating: Simulated_data/preprocessing/Simulated_dataset_2/metrics.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_2/._metrics.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_2/simulation_linear_ontrac_input.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_2/._simulation_linear_ontrac_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_2/simulated_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_2/simulated_expression.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_2/._simulated_expression.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_3/simulation_data_branch_non_linear_with_noise.rds  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_3/._simulation_data_branch_non_linear_with_noise.rds  
  inflating: Simulated_data/preprocessing/Simulated_dataset_3/metrics.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_3/._metrics.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_3/simulation_nonlinear_ontrac_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_3/simulated_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_3/simulated_expression.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_3/._simulated_expression.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_4/simulation_data_branch_disconnected_with_noise.rds  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_4/._simulation_data_branch_disconnected_with_noise.rds  
  inflating: Simulated_data/preprocessing/Simulated_dataset_4/simulation_disconnected_ontrac_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_4/metrics.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_4/._metrics.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_4/simulated_input.csv  
  inflating: Simulated_data/preprocessing/Simulated_dataset_4/simulated_expression.csv  
  inflating: __MACOSX/Simulated_data/preprocessing/Simulated_dataset_4/._simulated_expression.csv  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_16_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_5_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_5_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_1_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_100_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_1_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_5_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_base_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_base_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_1_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_16_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_base_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_base_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_3_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_base_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_base_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_100_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_hidden_feats_2_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_20_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_hidden_feats_2_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_6_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_2_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_base_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_base_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_5_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_100_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_k_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_k_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_1_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_16_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_gcn_layers_4_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_k_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_k_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_k_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_3_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_6_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_3_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_8_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_neighbors_20_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_hidden_feats_16_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_neighbors_20_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_hidden_feats_2_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_data_n_gcn_layers_6_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_gcn_layers_3_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_n_gcn_layers_6_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_disconnected_k_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_10_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_linear_n_neighbors_100_NN/processed/._pre_filter.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/processed/pre_transform.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/processed/._pre_transform.pt  
  inflating: Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/processed/pre_filter.pt  
  inflating: __MACOSX/Simulated_data/ONTraC_output/simulated_non_linear_n_neighbors_20_NN/processed/._pre_filter.pt  
```
:::

## ONTraC Run

```{code-cell}
:class: input-cell

ONTraC \
    --meta-input Simulated_data/preprocessing/Simulated_dataset_1/simulated_input.csv \
    --NN-dir output/simulation_NN \
    --GNN-dir output/simulation_GNN \
    --NT-dir output/simulation_NT \
    --device cuda \
    --epochs 1000 \
    --batch-size 5 \
    -s 42 \
    --patience 100 \
    --min-delta 0.001 \
    --min-epochs 50 \
    --lr 0.03 \
    --hidden-feats 4 \
    -k 6 \
    --modularity-loss-weight 0.3 \
    --regularization-loss-weight 0.1 \
    --purity-loss-weight 300 \
    --beta 0.03 2>&1 | tee log/simulation.log
```

:::{dropdown} Show output (Click to expand)
:class: folded-output

```text
##################################################################################

         ▄▄█▀▀██   ▀█▄   ▀█▀ █▀▀██▀▀█                   ▄▄█▀▀▀▄█
        ▄█▀    ██   █▀█   █     ██    ▄▄▄ ▄▄   ▄▄▄▄   ▄█▀     ▀
        ██      ██  █ ▀█▄ █     ██     ██▀ ▀▀ ▀▀ ▄██  ██
        ▀█▄     ██  █   ███     ██     ██     ▄█▀ ██  ▀█▄      ▄
         ▀▀█▄▄▄█▀  ▄█▄   ▀█    ▄██▄   ▄██▄    ▀█▄▄▀█▀  ▀▀█▄▄▄▄▀

                        version: 1.2.0

##################################################################################
16:10:00 --- INFO: ------------------ RUN params memo ------------------ 
16:10:00 --- INFO:             -------- I/O options -------             
16:10:00 --- INFO: Niche network output directory:  output/simulation_NN
16:10:00 --- INFO: GNN output directory:  output/simulation_GNN
16:10:00 --- INFO: Niche trajectory output directory:  output/simulation_NT
16:10:00 --- INFO: Meta data file:  Simulated_data/preprocessing/Simulated_dataset_1/simulated_input.csv
16:10:00 --- INFO:       -------- niche net constr options -------      
16:10:00 --- INFO: n_cpu:   4
16:10:00 --- INFO: n_neighbors: 50
16:10:00 --- INFO: n_local: 20
16:10:00 --- INFO:            -------- train options -------            
16:10:00 --- INFO: device:  cuda
16:10:00 --- INFO: epochs:  1000
16:10:00 --- INFO: batch_size:  5
16:10:00 --- INFO: patience:  100
16:10:00 --- INFO: min_delta:  0.001
16:10:00 --- INFO: min_epochs:  50
16:10:00 --- INFO: seed:  42
16:10:00 --- INFO: lr:  0.03
16:10:00 --- INFO: hidden_feats:  4
16:10:00 --- INFO: n_gcn_layers:  2
16:10:00 --- INFO: k:  6
16:10:00 --- INFO: modularity_loss_weight:  0.3
16:10:00 --- INFO: purity_loss_weight:  300.0
16:10:00 --- INFO: regularization_loss_weight:  0.1
16:10:00 --- INFO: beta:  0.03
16:10:00 --- INFO: ---------------- Niche trajectory options ----------------
16:10:00 --- INFO: Niche trajectory construction method: BF
16:10:00 --- INFO: --------------- RUN params memo end ----------------- 
16:10:00 --- INFO: ------------- Niche network construct --------------- 
16:10:00 --- INFO: Constructing niche network for sample: Simulation.
16:10:00 --- INFO: Building KNN network for sample: Simulation...
16:10:00 --- INFO: Calculating edge index for sample: Simulation...
16:10:00 --- INFO: Calculating niche weight matrix for sample: Simulation...
16:10:00 --- INFO: Calculating cell type composition for sample: Simulation...
16:10:00 --- INFO: Generating samples.yaml file.
16:10:00 --- INFO: ------------ Niche network construct end ------------ 
16:10:00 --- INFO: ------------------------ GNN ------------------------ 
16:10:00 --- INFO: Loading dataset.
16:10:00 --- INFO: Maximum number of cell in one sample is: 1100.
Processing...
16:10:01 --- INFO: Processing sample 1 of 1: Simulation
Done!
16:10:01 --- INFO: Processing sample 1 of 1: Simulation
16:10:03 --- INFO: epoch: 1, batch: 1, loss: 5.28548002243042, modularity_loss: -3.2737284527684096e-06, purity_loss: 5.216475963592529, regularization_loss: 0.06900753825902939
16:10:03 --- INFO: epoch: 2, batch: 1, loss: 5.285218715667725, modularity_loss: -6.284053142735502e-06, purity_loss: 5.216220855712891, regularization_loss: 0.06900409609079361
16:10:03 --- INFO: epoch: 3, batch: 1, loss: 5.284871578216553, modularity_loss: -1.0322082744096406e-05, purity_loss: 5.215879917144775, regularization_loss: 0.06900212913751602
16:10:03 --- INFO: epoch: 4, batch: 1, loss: 5.2844038009643555, modularity_loss: -1.5791869373060763e-05, purity_loss: 5.215417861938477, regularization_loss: 0.06900154799222946
16:10:03 --- INFO: epoch: 5, batch: 1, loss: 5.283760070800781, modularity_loss: -2.3352840798906982e-05, purity_loss: 5.214781284332275, regularization_loss: 0.06900208443403244
16:10:03 --- INFO: epoch: 6, batch: 1, loss: 5.282847881317139, modularity_loss: -3.4071570553351194e-05, purity_loss: 5.213878631591797, regularization_loss: 0.0690031498670578
16:10:03 --- INFO: epoch: 7, batch: 1, loss: 5.281559467315674, modularity_loss: -4.917501428280957e-05, purity_loss: 5.212604522705078, regularization_loss: 0.06900433450937271
16:10:03 --- INFO: epoch: 8, batch: 1, loss: 5.2797746658325195, modularity_loss: -7.003570499364287e-05, purity_loss: 5.210838794708252, regularization_loss: 0.06900576502084732
16:10:03 --- INFO: epoch: 9, batch: 1, loss: 5.277322292327881, modularity_loss: -9.865784522844478e-05, purity_loss: 5.208413124084473, regularization_loss: 0.0690079852938652
16:10:03 --- INFO: epoch: 10, batch: 1, loss: 5.273987770080566, modularity_loss: -0.00013753112580161542, purity_loss: 5.205113410949707, regularization_loss: 0.06901168823242188
16:10:03 --- INFO: epoch: 11, batch: 1, loss: 5.2695536613464355, modularity_loss: -0.00018925504991784692, purity_loss: 5.200725078582764, regularization_loss: 0.06901773065328598
16:10:03 --- INFO: epoch: 12, batch: 1, loss: 5.263756275177002, modularity_loss: -0.00025721127167344093, purity_loss: 5.194986343383789, regularization_loss: 0.06902701407670975
16:10:03 --- INFO: epoch: 13, batch: 1, loss: 5.256264686584473, modularity_loss: -0.00034563092049211264, purity_loss: 5.187570095062256, regularization_loss: 0.06904002279043198
16:10:03 --- INFO: epoch: 14, batch: 1, loss: 5.246661186218262, modularity_loss: -0.0004597530933097005, purity_loss: 5.178064346313477, regularization_loss: 0.06905638426542282
16:10:03 --- INFO: epoch: 15, batch: 1, loss: 5.234406471252441, modularity_loss: -0.0006063984474167228, purity_loss: 5.165937900543213, regularization_loss: 0.06907480210065842
16:10:03 --- INFO: epoch: 16, batch: 1, loss: 5.218887805938721, modularity_loss: -0.0007936400361359119, purity_loss: 5.150585651397705, regularization_loss: 0.0690956711769104
16:10:03 --- INFO: epoch: 17, batch: 1, loss: 5.199424743652344, modularity_loss: -0.0010309221688658, purity_loss: 5.1313347816467285, regularization_loss: 0.06912105530500412
16:10:03 --- INFO: epoch: 18, batch: 1, loss: 5.175188064575195, modularity_loss: -0.0013304809108376503, purity_loss: 5.107362747192383, regularization_loss: 0.06915583461523056
16:10:03 --- INFO: epoch: 19, batch: 1, loss: 5.145195960998535, modularity_loss: -0.0017075332580134273, purity_loss: 5.077697277069092, regularization_loss: 0.06920632719993591
16:10:03 --- INFO: epoch: 20, batch: 1, loss: 5.108408451080322, modularity_loss: -0.002179180271923542, purity_loss: 5.041309833526611, regularization_loss: 0.06927801668643951
16:10:03 --- INFO: epoch: 21, batch: 1, loss: 5.0638108253479, modularity_loss: -0.002763425000011921, purity_loss: 4.9972004890441895, regularization_loss: 0.06937380880117416
16:10:03 --- INFO: epoch: 22, batch: 1, loss: 5.01038932800293, modularity_loss: -0.0034794749226421118, purity_loss: 4.9443769454956055, regularization_loss: 0.06949187815189362
16:10:03 --- INFO: epoch: 23, batch: 1, loss: 4.947177886962891, modularity_loss: -0.004347739275544882, purity_loss: 4.881899356842041, regularization_loss: 0.06962606310844421
16:10:03 --- INFO: epoch: 24, batch: 1, loss: 4.873462200164795, modularity_loss: -0.00538914930075407, purity_loss: 4.809074878692627, regularization_loss: 0.06977648288011551
16:10:03 --- INFO: epoch: 25, batch: 1, loss: 4.78887414932251, modularity_loss: -0.006624677684158087, purity_loss: 4.725546360015869, regularization_loss: 0.06995254755020142
16:10:03 --- INFO: epoch: 26, batch: 1, loss: 4.693502426147461, modularity_loss: -0.008072642609477043, purity_loss: 4.631413459777832, regularization_loss: 0.07016138732433319
16:10:03 --- INFO: epoch: 27, batch: 1, loss: 4.588165283203125, modularity_loss: -0.009743880480527878, purity_loss: 4.527505397796631, regularization_loss: 0.07040389627218246
16:10:03 --- INFO: epoch: 28, batch: 1, loss: 4.474507808685303, modularity_loss: -0.011639253236353397, purity_loss: 4.4154744148254395, regularization_loss: 0.07067283242940903
16:10:03 --- INFO: epoch: 29, batch: 1, loss: 4.355066299438477, modularity_loss: -0.013746660202741623, purity_loss: 4.297853469848633, regularization_loss: 0.07095951586961746
16:10:03 --- INFO: epoch: 30, batch: 1, loss: 4.233127593994141, modularity_loss: -0.016040191054344177, purity_loss: 4.177913665771484, regularization_loss: 0.07125414162874222
16:10:03 --- INFO: epoch: 31, batch: 1, loss: 4.112618446350098, modularity_loss: -0.018478065729141235, purity_loss: 4.059542655944824, regularization_loss: 0.07155390083789825
16:10:03 --- INFO: epoch: 32, batch: 1, loss: 3.9976160526275635, modularity_loss: -0.021002966910600662, purity_loss: 3.946774482727051, regularization_loss: 0.07184451073408127
16:10:03 --- INFO: epoch: 33, batch: 1, loss: 3.891801357269287, modularity_loss: -0.02354886196553707, purity_loss: 3.843238115310669, regularization_loss: 0.07211221754550934
16:10:03 --- INFO: epoch: 34, batch: 1, loss: 3.7979230880737305, modularity_loss: -0.026050761342048645, purity_loss: 3.751629590988159, regularization_loss: 0.07234418392181396
16:10:03 --- INFO: epoch: 35, batch: 1, loss: 3.7175583839416504, modularity_loss: -0.028448864817619324, purity_loss: 3.673473596572876, regularization_loss: 0.07253360003232956
16:10:03 --- INFO: epoch: 36, batch: 1, loss: 3.6510701179504395, modularity_loss: -0.03069290705025196, purity_loss: 3.6090734004974365, regularization_loss: 0.07268955558538437
16:10:03 --- INFO: epoch: 37, batch: 1, loss: 3.597705125808716, modularity_loss: -0.03274816274642944, purity_loss: 3.557629346847534, regularization_loss: 0.07282402366399765
16:10:03 --- INFO: epoch: 38, batch: 1, loss: 3.555978536605835, modularity_loss: -0.03459617868065834, purity_loss: 3.517627477645874, regularization_loss: 0.07294731587171555
16:10:03 --- INFO: epoch: 39, batch: 1, loss: 3.5240330696105957, modularity_loss: -0.03623472899198532, purity_loss: 3.487198829650879, regularization_loss: 0.07306908071041107
16:10:03 --- INFO: epoch: 40, batch: 1, loss: 3.499948263168335, modularity_loss: -0.03767409548163414, purity_loss: 3.4644253253936768, regularization_loss: 0.0731971487402916
16:10:03 --- INFO: epoch: 41, batch: 1, loss: 3.48195219039917, modularity_loss: -0.038934800773859024, purity_loss: 3.4475505352020264, regularization_loss: 0.07333650439977646
16:10:03 --- INFO: epoch: 42, batch: 1, loss: 3.468569755554199, modularity_loss: -0.04003976657986641, purity_loss: 3.435119390487671, regularization_loss: 0.07349014282226562
16:10:03 --- INFO: epoch: 43, batch: 1, loss: 3.4585952758789062, modularity_loss: -0.041014414280653, purity_loss: 3.4259495735168457, regularization_loss: 0.0736600011587143
16:10:03 --- INFO: epoch: 44, batch: 1, loss: 3.451097011566162, modularity_loss: -0.04188314080238342, purity_loss: 3.419132709503174, regularization_loss: 0.07384748756885529
16:10:03 --- INFO: epoch: 45, batch: 1, loss: 3.4453680515289307, modularity_loss: -0.04266894608736038, purity_loss: 3.413982629776001, regularization_loss: 0.07405431568622589
16:10:04 --- INFO: epoch: 46, batch: 1, loss: 3.4408679008483887, modularity_loss: -0.04339265078306198, purity_loss: 3.409978151321411, regularization_loss: 0.07428240031003952
16:10:04 --- INFO: epoch: 47, batch: 1, loss: 3.4371962547302246, modularity_loss: -0.044073544442653656, purity_loss: 3.406735897064209, regularization_loss: 0.07453381270170212
16:10:04 --- INFO: epoch: 48, batch: 1, loss: 3.434048652648926, modularity_loss: -0.04472927004098892, purity_loss: 3.4039666652679443, regularization_loss: 0.07481134682893753
16:10:04 --- INFO: epoch: 49, batch: 1, loss: 3.431185722351074, modularity_loss: -0.045375436544418335, purity_loss: 3.4014430046081543, regularization_loss: 0.07511819899082184
16:10:04 --- INFO: epoch: 50, batch: 1, loss: 3.428419351577759, modularity_loss: -0.04602745547890663, purity_loss: 3.3989882469177246, regularization_loss: 0.0754585713148117
16:10:04 --- INFO: epoch: 51, batch: 1, loss: 3.4255855083465576, modularity_loss: -0.0466998927295208, purity_loss: 3.3964476585388184, regularization_loss: 0.075837641954422
16:10:04 --- INFO: epoch: 52, batch: 1, loss: 3.4225478172302246, modularity_loss: -0.04740772396326065, purity_loss: 3.393693685531616, regularization_loss: 0.076261967420578
16:10:04 --- INFO: epoch: 53, batch: 1, loss: 3.4191815853118896, modularity_loss: -0.04816614091396332, purity_loss: 3.390608549118042, regularization_loss: 0.07673914730548859
16:10:04 --- INFO: epoch: 54, batch: 1, loss: 3.415365219116211, modularity_loss: -0.048991184681653976, purity_loss: 3.387078285217285, regularization_loss: 0.07727809995412827
16:10:04 --- INFO: epoch: 55, batch: 1, loss: 3.410975933074951, modularity_loss: -0.0499001145362854, purity_loss: 3.3829872608184814, regularization_loss: 0.07788889110088348
16:10:04 --- INFO: epoch: 56, batch: 1, loss: 3.4058806896209717, modularity_loss: -0.050910744816064835, purity_loss: 3.378209114074707, regularization_loss: 0.0785822719335556
16:10:04 --- INFO: epoch: 57, batch: 1, loss: 3.3999404907226562, modularity_loss: -0.05204174667596817, purity_loss: 3.372612714767456, regularization_loss: 0.07936952263116837
16:10:04 --- INFO: epoch: 58, batch: 1, loss: 3.393017292022705, modularity_loss: -0.05330994352698326, purity_loss: 3.3660669326782227, regularization_loss: 0.08026034384965897
16:10:04 --- INFO: epoch: 59, batch: 1, loss: 3.384976625442505, modularity_loss: -0.05472927168011665, purity_loss: 3.3584437370300293, regularization_loss: 0.08126217871904373
16:10:04 --- INFO: epoch: 60, batch: 1, loss: 3.3756861686706543, modularity_loss: -0.056307047605514526, purity_loss: 3.3496153354644775, regularization_loss: 0.0823778584599495
16:10:04 --- INFO: epoch: 61, batch: 1, loss: 3.3650622367858887, modularity_loss: -0.058039724826812744, purity_loss: 3.3395004272460938, regularization_loss: 0.08360160142183304
16:10:04 --- INFO: epoch: 62, batch: 1, loss: 3.3530921936035156, modularity_loss: -0.05990936979651451, purity_loss: 3.3280861377716064, regularization_loss: 0.08491552621126175
16:10:04 --- INFO: epoch: 63, batch: 1, loss: 3.3398354053497314, modularity_loss: -0.061880651861429214, purity_loss: 3.315425395965576, regularization_loss: 0.08629056811332703
16:10:04 --- INFO: epoch: 64, batch: 1, loss: 3.3254451751708984, modularity_loss: -0.06390120834112167, purity_loss: 3.3016655445098877, regularization_loss: 0.087680883705616
16:10:04 --- INFO: epoch: 65, batch: 1, loss: 3.3101494312286377, modularity_loss: -0.0659104734659195, purity_loss: 3.2870290279388428, regularization_loss: 0.08903095871210098
16:10:04 --- INFO: epoch: 66, batch: 1, loss: 3.2942137718200684, modularity_loss: -0.06784937530755997, purity_loss: 3.271775007247925, regularization_loss: 0.09028825908899307
16:10:04 --- INFO: epoch: 67, batch: 1, loss: 3.2778971195220947, modularity_loss: -0.06966306269168854, purity_loss: 3.2561533451080322, regularization_loss: 0.09140689671039581
16:10:04 --- INFO: epoch: 68, batch: 1, loss: 3.261399030685425, modularity_loss: -0.07131224870681763, purity_loss: 3.240354061126709, regularization_loss: 0.09235730767250061
16:10:04 --- INFO: epoch: 69, batch: 1, loss: 3.2448348999023438, modularity_loss: -0.07277137041091919, purity_loss: 3.224482297897339, regularization_loss: 0.09312398731708527
16:10:04 --- INFO: epoch: 70, batch: 1, loss: 3.2282180786132812, modularity_loss: -0.07402641326189041, purity_loss: 3.208543062210083, regularization_loss: 0.09370134770870209
16:10:04 --- INFO: epoch: 71, batch: 1, loss: 3.2113988399505615, modularity_loss: -0.07506338506937027, purity_loss: 3.1923770904541016, regularization_loss: 0.09408514946699142
16:10:04 --- INFO: epoch: 72, batch: 1, loss: 3.1942474842071533, modularity_loss: -0.07587370276451111, purity_loss: 3.1758460998535156, regularization_loss: 0.09427505731582642
16:10:04 --- INFO: epoch: 73, batch: 1, loss: 3.1766467094421387, modularity_loss: -0.076450414955616, purity_loss: 3.158820867538452, regularization_loss: 0.09427613765001297
16:10:04 --- INFO: epoch: 74, batch: 1, loss: 3.1584489345550537, modularity_loss: -0.07678649574518204, purity_loss: 3.141144275665283, regularization_loss: 0.09409104287624359
16:10:04 --- INFO: epoch: 75, batch: 1, loss: 3.139439582824707, modularity_loss: -0.07689649611711502, purity_loss: 3.1225948333740234, regularization_loss: 0.09374135732650757
16:10:04 --- INFO: epoch: 76, batch: 1, loss: 3.1194751262664795, modularity_loss: -0.07682538032531738, purity_loss: 3.1030330657958984, regularization_loss: 0.09326745569705963
16:10:04 --- INFO: epoch: 77, batch: 1, loss: 3.098288059234619, modularity_loss: -0.07660239189863205, purity_loss: 3.0821988582611084, regularization_loss: 0.09269153326749802
16:10:04 --- INFO: epoch: 78, batch: 1, loss: 3.0757675170898438, modularity_loss: -0.07624781876802444, purity_loss: 3.0599911212921143, regularization_loss: 0.09202416986227036
16:10:04 --- INFO: epoch: 79, batch: 1, loss: 3.051815986633301, modularity_loss: -0.07578045129776001, purity_loss: 3.0363197326660156, regularization_loss: 0.09127674996852875
16:10:04 --- INFO: epoch: 80, batch: 1, loss: 3.026252031326294, modularity_loss: -0.07522313296794891, purity_loss: 3.0110132694244385, regularization_loss: 0.09046197682619095
16:10:04 --- INFO: epoch: 81, batch: 1, loss: 2.9988222122192383, modularity_loss: -0.07459431886672974, purity_loss: 2.9838297367095947, regularization_loss: 0.08958683907985687
16:10:04 --- INFO: epoch: 82, batch: 1, loss: 2.9692039489746094, modularity_loss: -0.07390538603067398, purity_loss: 2.9544565677642822, regularization_loss: 0.08865278959274292
16:10:04 --- INFO: epoch: 83, batch: 1, loss: 2.9366180896759033, modularity_loss: -0.07313193380832672, purity_loss: 2.922114372253418, regularization_loss: 0.08763554692268372
16:10:04 --- INFO: epoch: 84, batch: 1, loss: 2.9003798961639404, modularity_loss: -0.07225389778614044, purity_loss: 2.886115074157715, regularization_loss: 0.0865187868475914
16:10:04 --- INFO: epoch: 85, batch: 1, loss: 2.8596553802490234, modularity_loss: -0.0712897852063179, purity_loss: 2.8456289768218994, regularization_loss: 0.08531614392995834
16:10:04 --- INFO: epoch: 86, batch: 1, loss: 2.813706159591675, modularity_loss: -0.07027032226324081, purity_loss: 2.799922466278076, regularization_loss: 0.0840541273355484
16:10:04 --- INFO: epoch: 87, batch: 1, loss: 2.7620697021484375, modularity_loss: -0.06924298405647278, purity_loss: 2.7485363483428955, regularization_loss: 0.08277627825737
16:10:04 --- INFO: epoch: 88, batch: 1, loss: 2.704533338546753, modularity_loss: -0.0682893916964531, purity_loss: 2.691272020339966, regularization_loss: 0.08155069500207901
16:10:04 --- INFO: epoch: 89, batch: 1, loss: 2.6416075229644775, modularity_loss: -0.06755015254020691, purity_loss: 2.628682851791382, regularization_loss: 0.08047474920749664
16:10:04 --- INFO: epoch: 90, batch: 1, loss: 2.574474811553955, modularity_loss: -0.06721139699220657, purity_loss: 2.5620388984680176, regularization_loss: 0.0796472579240799
16:10:04 --- INFO: epoch: 91, batch: 1, loss: 2.5048861503601074, modularity_loss: -0.06742584705352783, purity_loss: 2.493195056915283, regularization_loss: 0.07911687344312668
16:10:04 --- INFO: epoch: 92, batch: 1, loss: 2.4356489181518555, modularity_loss: -0.06825752556324005, purity_loss: 2.4250400066375732, regularization_loss: 0.07886631786823273
16:10:04 --- INFO: epoch: 93, batch: 1, loss: 2.3702807426452637, modularity_loss: -0.06959864497184753, purity_loss: 2.361046552658081, regularization_loss: 0.07883286476135254
16:10:04 --- INFO: epoch: 94, batch: 1, loss: 2.3123087882995605, modularity_loss: -0.07127038389444351, purity_loss: 2.304588794708252, regularization_loss: 0.07899042218923569
16:10:04 --- INFO: epoch: 95, batch: 1, loss: 2.263590097427368, modularity_loss: -0.07314881682395935, purity_loss: 2.2573938369750977, regularization_loss: 0.07934505492448807
16:10:04 --- INFO: epoch: 96, batch: 1, loss: 2.224318742752075, modularity_loss: -0.07517903298139572, purity_loss: 2.2195966243743896, regularization_loss: 0.07990123331546783
16:10:04 --- INFO: epoch: 97, batch: 1, loss: 2.193443536758423, modularity_loss: -0.07732480764389038, purity_loss: 2.190141439437866, regularization_loss: 0.08062697947025299
16:10:04 --- INFO: epoch: 98, batch: 1, loss: 2.169196367263794, modularity_loss: -0.07953200489282608, purity_loss: 2.167280673980713, regularization_loss: 0.08144767582416534
16:10:04 --- INFO: epoch: 99, batch: 1, loss: 2.150085926055908, modularity_loss: -0.08170461654663086, purity_loss: 2.149509906768799, regularization_loss: 0.08228059858083725
16:10:04 --- INFO: epoch: 100, batch: 1, loss: 2.1345691680908203, modularity_loss: -0.0837986171245575, purity_loss: 2.1353137493133545, regularization_loss: 0.08305415511131287
16:10:04 --- INFO: epoch: 101, batch: 1, loss: 2.121586561203003, modularity_loss: -0.08576671779155731, purity_loss: 2.123623847961426, regularization_loss: 0.0837293267250061
16:10:04 --- INFO: epoch: 102, batch: 1, loss: 2.110600709915161, modularity_loss: -0.08758178353309631, purity_loss: 2.1138761043548584, regularization_loss: 0.08430635929107666
16:10:04 --- INFO: epoch: 103, batch: 1, loss: 2.101208209991455, modularity_loss: -0.08925171196460724, purity_loss: 2.105654239654541, regularization_loss: 0.08480566740036011
16:10:04 --- INFO: epoch: 104, batch: 1, loss: 2.09316086769104, modularity_loss: -0.09080300480127335, purity_loss: 2.098705530166626, regularization_loss: 0.08525842428207397
16:10:04 --- INFO: epoch: 105, batch: 1, loss: 2.08624267578125, modularity_loss: -0.09226067364215851, purity_loss: 2.0928072929382324, regularization_loss: 0.0856960117816925
16:10:04 --- INFO: epoch: 106, batch: 1, loss: 2.0801608562469482, modularity_loss: -0.09365560114383698, purity_loss: 2.0876731872558594, regularization_loss: 0.08614324033260345
16:10:04 --- INFO: epoch: 107, batch: 1, loss: 2.074697256088257, modularity_loss: -0.09500505775213242, purity_loss: 2.0830905437469482, regularization_loss: 0.08661186695098877
16:10:04 --- INFO: epoch: 108, batch: 1, loss: 2.0697340965270996, modularity_loss: -0.09631598740816116, purity_loss: 2.0789473056793213, regularization_loss: 0.0871027484536171
16:10:04 --- INFO: epoch: 109, batch: 1, loss: 2.065192937850952, modularity_loss: -0.09758702665567398, purity_loss: 2.0751700401306152, regularization_loss: 0.08760981261730194
16:10:04 --- INFO: epoch: 110, batch: 1, loss: 2.0610570907592773, modularity_loss: -0.09880737960338593, purity_loss: 2.0717430114746094, regularization_loss: 0.08812142163515091
16:10:04 --- INFO: epoch: 111, batch: 1, loss: 2.0573222637176514, modularity_loss: -0.09996522217988968, purity_loss: 2.0686628818511963, regularization_loss: 0.08862455189228058
16:10:04 --- INFO: epoch: 112, batch: 1, loss: 2.0539493560791016, modularity_loss: -0.10105035454034805, purity_loss: 2.0658936500549316, regularization_loss: 0.08910609036684036
16:10:04 --- INFO: epoch: 113, batch: 1, loss: 2.0508639812469482, modularity_loss: -0.10205364227294922, purity_loss: 2.0633645057678223, regularization_loss: 0.08955316990613937
16:10:04 --- INFO: epoch: 114, batch: 1, loss: 2.048006057739258, modularity_loss: -0.10297368466854095, purity_loss: 2.061021327972412, regularization_loss: 0.08995833247900009
16:10:04 --- INFO: epoch: 115, batch: 1, loss: 2.045267343521118, modularity_loss: -0.10381319373846054, purity_loss: 2.058762311935425, regularization_loss: 0.09031816571950912
16:10:04 --- INFO: epoch: 116, batch: 1, loss: 2.042548179626465, modularity_loss: -0.10457626730203629, purity_loss: 2.0564920902252197, regularization_loss: 0.09063225984573364
16:10:04 --- INFO: epoch: 117, batch: 1, loss: 2.039834499359131, modularity_loss: -0.10527132451534271, purity_loss: 2.0542004108428955, regularization_loss: 0.09090538322925568
16:10:04 --- INFO: epoch: 118, batch: 1, loss: 2.0370523929595947, modularity_loss: -0.10590749979019165, purity_loss: 2.0518150329589844, regularization_loss: 0.09114475548267365
16:10:04 --- INFO: epoch: 119, batch: 1, loss: 2.0341577529907227, modularity_loss: -0.10649248212575912, purity_loss: 2.04929256439209, regularization_loss: 0.09135771542787552
16:10:04 --- INFO: epoch: 120, batch: 1, loss: 2.0311119556427, modularity_loss: -0.10703201591968536, purity_loss: 2.046593189239502, regularization_loss: 0.091550812125206
16:10:04 --- INFO: epoch: 121, batch: 1, loss: 2.027860641479492, modularity_loss: -0.10752977430820465, purity_loss: 2.043661594390869, regularization_loss: 0.09172891825437546
16:10:04 --- INFO: epoch: 122, batch: 1, loss: 2.024348020553589, modularity_loss: -0.10798539966344833, purity_loss: 2.040440320968628, regularization_loss: 0.09189317375421524
16:10:04 --- INFO: epoch: 123, batch: 1, loss: 2.0205211639404297, modularity_loss: -0.1083960011601448, purity_loss: 2.0368757247924805, regularization_loss: 0.09204134345054626
16:10:04 --- INFO: epoch: 124, batch: 1, loss: 2.0163164138793945, modularity_loss: -0.10875429958105087, purity_loss: 2.032902479171753, regularization_loss: 0.09216825664043427
16:10:04 --- INFO: epoch: 125, batch: 1, loss: 2.0116453170776367, modularity_loss: -0.10905242711305618, purity_loss: 2.0284299850463867, regularization_loss: 0.0922677218914032
16:10:04 --- INFO: epoch: 126, batch: 1, loss: 2.006394624710083, modularity_loss: -0.10927897691726685, purity_loss: 2.023343324661255, regularization_loss: 0.09233031421899796
16:10:04 --- INFO: epoch: 127, batch: 1, loss: 2.000448703765869, modularity_loss: -0.10942075401544571, purity_loss: 2.017524480819702, regularization_loss: 0.09234504401683807
16:10:04 --- INFO: epoch: 128, batch: 1, loss: 1.9936673641204834, modularity_loss: -0.10946736484766006, purity_loss: 2.010831356048584, regularization_loss: 0.09230338782072067
16:10:04 --- INFO: epoch: 129, batch: 1, loss: 1.9858568906784058, modularity_loss: -0.10940638184547424, purity_loss: 2.0030672550201416, regularization_loss: 0.09219598025083542
16:10:04 --- INFO: epoch: 130, batch: 1, loss: 1.9768791198730469, modularity_loss: -0.10922740399837494, purity_loss: 1.994091510772705, regularization_loss: 0.09201504290103912
16:10:04 --- INFO: epoch: 131, batch: 1, loss: 1.966536045074463, modularity_loss: -0.1089196503162384, purity_loss: 1.9837021827697754, regularization_loss: 0.09175354242324829
16:10:04 --- INFO: epoch: 132, batch: 1, loss: 1.954637050628662, modularity_loss: -0.10847203433513641, purity_loss: 1.971704363822937, regularization_loss: 0.09140466153621674
16:10:04 --- INFO: epoch: 133, batch: 1, loss: 1.9409703016281128, modularity_loss: -0.1078687310218811, purity_loss: 1.9578789472579956, regularization_loss: 0.09096008539199829
16:10:04 --- INFO: epoch: 134, batch: 1, loss: 1.925367832183838, modularity_loss: -0.10709644109010696, purity_loss: 1.9420515298843384, regularization_loss: 0.0904126912355423
16:10:04 --- INFO: epoch: 135, batch: 1, loss: 1.9077285528182983, modularity_loss: -0.10613854974508286, purity_loss: 1.9241135120391846, regularization_loss: 0.08975359052419662
16:10:04 --- INFO: epoch: 136, batch: 1, loss: 1.8881088495254517, modularity_loss: -0.10498607158660889, purity_loss: 1.9041131734848022, regularization_loss: 0.08898171782493591
16:10:04 --- INFO: epoch: 137, batch: 1, loss: 1.8666999340057373, modularity_loss: -0.10364474356174469, purity_loss: 1.8822377920150757, regularization_loss: 0.08810693770647049
16:10:04 --- INFO: epoch: 138, batch: 1, loss: 1.8438925743103027, modularity_loss: -0.10214832425117493, purity_loss: 1.8588824272155762, regularization_loss: 0.08715841919183731
16:10:04 --- INFO: epoch: 139, batch: 1, loss: 1.8202645778656006, modularity_loss: -0.10057496279478073, purity_loss: 1.8346441984176636, regularization_loss: 0.08619533479213715
16:10:04 --- INFO: epoch: 140, batch: 1, loss: 1.7964093685150146, modularity_loss: -0.09905555844306946, purity_loss: 1.8101575374603271, regularization_loss: 0.08530744165182114
16:10:04 --- INFO: epoch: 141, batch: 1, loss: 1.7727038860321045, modularity_loss: -0.09775801002979279, purity_loss: 1.7858649492263794, regularization_loss: 0.0845969095826149
16:10:04 --- INFO: epoch: 142, batch: 1, loss: 1.7490986585617065, modularity_loss: -0.09684277325868607, purity_loss: 1.7617995738983154, regularization_loss: 0.08414185047149658
16:10:04 --- INFO: epoch: 143, batch: 1, loss: 1.7251338958740234, modularity_loss: -0.09641066938638687, purity_loss: 1.7375839948654175, regularization_loss: 0.08396057039499283
16:10:04 --- INFO: epoch: 144, batch: 1, loss: 1.7002218961715698, modularity_loss: -0.0964755043387413, purity_loss: 1.7126898765563965, regularization_loss: 0.08400757610797882
16:10:04 --- INFO: epoch: 145, batch: 1, loss: 1.6740801334381104, modularity_loss: -0.0969858318567276, purity_loss: 1.686849594116211, regularization_loss: 0.08421637117862701
16:10:04 --- INFO: epoch: 146, batch: 1, loss: 1.6473498344421387, modularity_loss: -0.09784870594739914, purity_loss: 1.6606656312942505, regularization_loss: 0.08453290909528732
16:10:04 --- INFO: epoch: 147, batch: 1, loss: 1.6212257146835327, modularity_loss: -0.0989590436220169, purity_loss: 1.6352561712265015, regularization_loss: 0.08492863178253174
16:10:04 --- INFO: epoch: 148, batch: 1, loss: 1.5968538522720337, modularity_loss: -0.10020986944437027, purity_loss: 1.6116693019866943, regularization_loss: 0.08539439737796783
16:10:04 --- INFO: epoch: 149, batch: 1, loss: 1.5749542713165283, modularity_loss: -0.10150553286075592, purity_loss: 1.5905368328094482, regularization_loss: 0.08592300862073898
16:10:04 --- INFO: epoch: 150, batch: 1, loss: 1.5556694269180298, modularity_loss: -0.10277212411165237, purity_loss: 1.5719490051269531, regularization_loss: 0.08649258315563202
16:10:05 --- INFO: epoch: 151, batch: 1, loss: 1.5385600328445435, modularity_loss: -0.10396552830934525, purity_loss: 1.5554530620574951, regularization_loss: 0.08707251399755478
16:10:05 --- INFO: epoch: 152, batch: 1, loss: 1.5232881307601929, modularity_loss: -0.10506773740053177, purity_loss: 1.5407260656356812, regularization_loss: 0.0876297578215599
16:10:05 --- INFO: epoch: 153, batch: 1, loss: 1.5095598697662354, modularity_loss: -0.10608130693435669, purity_loss: 1.5275046825408936, regularization_loss: 0.08813653886318207
16:10:05 --- INFO: epoch: 154, batch: 1, loss: 1.4972636699676514, modularity_loss: -0.10701730102300644, purity_loss: 1.5157074928283691, regularization_loss: 0.08857347071170807
16:10:05 --- INFO: epoch: 155, batch: 1, loss: 1.4861924648284912, modularity_loss: -0.10788371413946152, purity_loss: 1.5051482915878296, regularization_loss: 0.08892782777547836
16:10:05 --- INFO: epoch: 156, batch: 1, loss: 1.4762133359909058, modularity_loss: -0.1086820736527443, purity_loss: 1.49569833278656, regularization_loss: 0.08919712901115417
16:10:05 --- INFO: epoch: 157, batch: 1, loss: 1.4673134088516235, modularity_loss: -0.10940898954868317, purity_loss: 1.4873325824737549, regularization_loss: 0.08938977122306824
16:10:05 --- INFO: epoch: 158, batch: 1, loss: 1.459471583366394, modularity_loss: -0.11006661504507065, purity_loss: 1.4800143241882324, regularization_loss: 0.08952391892671585
16:10:05 --- INFO: epoch: 159, batch: 1, loss: 1.452507734298706, modularity_loss: -0.11066672950983047, purity_loss: 1.473552942276001, regularization_loss: 0.08962152153253555
16:10:05 --- INFO: epoch: 160, batch: 1, loss: 1.446143627166748, modularity_loss: -0.1112254336476326, purity_loss: 1.4676662683486938, regularization_loss: 0.08970282226800919
16:10:05 --- INFO: epoch: 161, batch: 1, loss: 1.4401370286941528, modularity_loss: -0.11175957322120667, purity_loss: 1.4621132612228394, regularization_loss: 0.08978335559368134
16:10:05 --- INFO: epoch: 162, batch: 1, loss: 1.434458613395691, modularity_loss: -0.11227940768003464, purity_loss: 1.4568651914596558, regularization_loss: 0.08987283706665039
16:10:05 --- INFO: epoch: 163, batch: 1, loss: 1.4292229413986206, modularity_loss: -0.11278869956731796, purity_loss: 1.452038049697876, regularization_loss: 0.08997362107038498
16:10:05 --- INFO: epoch: 164, batch: 1, loss: 1.42454195022583, modularity_loss: -0.11328516900539398, purity_loss: 1.4477461576461792, regularization_loss: 0.09008099138736725
16:10:05 --- INFO: epoch: 165, batch: 1, loss: 1.4204002618789673, modularity_loss: -0.11376234889030457, purity_loss: 1.4439774751663208, regularization_loss: 0.09018510580062866
16:10:05 --- INFO: epoch: 166, batch: 1, loss: 1.4166680574417114, modularity_loss: -0.11421302706003189, purity_loss: 1.4406052827835083, regularization_loss: 0.09027580171823502
16:10:05 --- INFO: epoch: 167, batch: 1, loss: 1.4132226705551147, modularity_loss: -0.11462980508804321, purity_loss: 1.4375063180923462, regularization_loss: 0.09034613519906998
16:10:05 --- INFO: epoch: 168, batch: 1, loss: 1.4100229740142822, modularity_loss: -0.11500607430934906, purity_loss: 1.434635043144226, regularization_loss: 0.09039401262998581
16:10:05 --- INFO: epoch: 169, batch: 1, loss: 1.4070929288864136, modularity_loss: -0.11533790081739426, purity_loss: 1.4320088624954224, regularization_loss: 0.09042193740606308
16:10:05 --- INFO: epoch: 170, batch: 1, loss: 1.4044541120529175, modularity_loss: -0.11562607437372208, purity_loss: 1.4296448230743408, regularization_loss: 0.09043541550636292
16:10:05 --- INFO: epoch: 171, batch: 1, loss: 1.4020979404449463, modularity_loss: -0.11587591469287872, purity_loss: 1.4275329113006592, regularization_loss: 0.09044100344181061
16:10:05 --- INFO: epoch: 172, batch: 1, loss: 1.399997591972351, modularity_loss: -0.11609559506177902, purity_loss: 1.4256486892700195, regularization_loss: 0.09044446051120758
16:10:05 --- INFO: epoch: 173, batch: 1, loss: 1.398127794265747, modularity_loss: -0.11629407107830048, purity_loss: 1.423972487449646, regularization_loss: 0.09044931828975677
16:10:05 --- INFO: epoch: 174, batch: 1, loss: 1.3964709043502808, modularity_loss: -0.11647879332304001, purity_loss: 1.4224929809570312, regularization_loss: 0.09045669436454773
16:10:05 --- INFO: epoch: 175, batch: 1, loss: 1.3949960470199585, modularity_loss: -0.11665524542331696, purity_loss: 1.4211856126785278, regularization_loss: 0.09046570211648941
16:10:05 --- INFO: epoch: 176, batch: 1, loss: 1.393661618232727, modularity_loss: -0.11682667583227158, purity_loss: 1.4200137853622437, regularization_loss: 0.09047455340623856
16:10:05 --- INFO: epoch: 177, batch: 1, loss: 1.3924314975738525, modularity_loss: -0.1169944480061531, purity_loss: 1.418944001197815, regularization_loss: 0.09048197418451309
16:10:05 --- INFO: epoch: 178, batch: 1, loss: 1.3913006782531738, modularity_loss: -0.11715778708457947, purity_loss: 1.4179706573486328, regularization_loss: 0.09048780798912048
16:10:05 --- INFO: epoch: 179, batch: 1, loss: 1.3902850151062012, modularity_loss: -0.11731407046318054, purity_loss: 1.4171063899993896, regularization_loss: 0.09049274772405624
16:10:05 --- INFO: epoch: 180, batch: 1, loss: 1.3893944025039673, modularity_loss: -0.11745989322662354, purity_loss: 1.4163565635681152, regularization_loss: 0.09049775451421738
16:10:05 --- INFO: epoch: 181, batch: 1, loss: 1.3886122703552246, modularity_loss: -0.1175922304391861, purity_loss: 1.4157010316848755, regularization_loss: 0.09050344675779343
16:10:05 --- INFO: epoch: 182, batch: 1, loss: 1.387902021408081, modularity_loss: -0.11770882457494736, purity_loss: 1.4151009321212769, regularization_loss: 0.09050992131233215
16:10:05 --- INFO: epoch: 183, batch: 1, loss: 1.3872311115264893, modularity_loss: -0.11780886352062225, purity_loss: 1.4145231246948242, regularization_loss: 0.09051679819822311
16:10:05 --- INFO: epoch: 184, batch: 1, loss: 1.3865885734558105, modularity_loss: -0.11789318174123764, purity_loss: 1.4139580726623535, regularization_loss: 0.09052368253469467
16:10:05 --- INFO: epoch: 185, batch: 1, loss: 1.3859813213348389, modularity_loss: -0.11796402931213379, purity_loss: 1.4134153127670288, regularization_loss: 0.09053007513284683
16:10:05 --- INFO: epoch: 186, batch: 1, loss: 1.385420799255371, modularity_loss: -0.11802443116903305, purity_loss: 1.4129096269607544, regularization_loss: 0.09053562581539154
16:10:05 --- INFO: epoch: 187, batch: 1, loss: 1.3849077224731445, modularity_loss: -0.11807774752378464, purity_loss: 1.4124455451965332, regularization_loss: 0.09053990244865417
16:10:05 --- INFO: epoch: 188, batch: 1, loss: 1.3844327926635742, modularity_loss: -0.11812686920166016, purity_loss: 1.4120169878005981, regularization_loss: 0.09054265916347504
16:10:05 --- INFO: epoch: 189, batch: 1, loss: 1.3839815855026245, modularity_loss: -0.11817411333322525, purity_loss: 1.4116116762161255, regularization_loss: 0.09054405242204666
16:10:05 --- INFO: epoch: 190, batch: 1, loss: 1.3835468292236328, modularity_loss: -0.11822064965963364, purity_loss: 1.4112228155136108, regularization_loss: 0.09054472297430038
16:10:05 --- INFO: epoch: 191, batch: 1, loss: 1.3831254243850708, modularity_loss: -0.11826681345701218, purity_loss: 1.4108468294143677, regularization_loss: 0.09054543077945709
16:10:05 --- INFO: epoch: 192, batch: 1, loss: 1.3827180862426758, modularity_loss: -0.1183122992515564, purity_loss: 1.4104833602905273, regularization_loss: 0.09054702520370483
16:10:05 --- INFO: epoch: 193, batch: 1, loss: 1.3823237419128418, modularity_loss: -0.11835602670907974, purity_loss: 1.4101299047470093, regularization_loss: 0.09054983407258987
16:10:05 --- INFO: epoch: 194, batch: 1, loss: 1.381941318511963, modularity_loss: -0.11839670687913895, purity_loss: 1.4097844362258911, regularization_loss: 0.09055359661579132
16:10:05 --- INFO: epoch: 195, batch: 1, loss: 1.3815701007843018, modularity_loss: -0.11843288689851761, purity_loss: 1.4094454050064087, regularization_loss: 0.0905575305223465
16:10:05 --- INFO: epoch: 196, batch: 1, loss: 1.3812094926834106, modularity_loss: -0.11846328526735306, purity_loss: 1.4091122150421143, regularization_loss: 0.09056055545806885
16:10:05 --- INFO: epoch: 197, batch: 1, loss: 1.3808577060699463, modularity_loss: -0.11848718672990799, purity_loss: 1.4087834358215332, regularization_loss: 0.09056145697832108
16:10:05 --- INFO: epoch: 198, batch: 1, loss: 1.380511999130249, modularity_loss: -0.1185043528676033, purity_loss: 1.4084569215774536, regularization_loss: 0.09055939316749573
16:10:05 --- INFO: epoch: 199, batch: 1, loss: 1.3801681995391846, modularity_loss: -0.11851521581411362, purity_loss: 1.4081294536590576, regularization_loss: 0.09055400639772415
16:10:05 --- INFO: epoch: 200, batch: 1, loss: 1.379824161529541, modularity_loss: -0.118520587682724, purity_loss: 1.4077991247177124, regularization_loss: 0.09054556488990784
16:10:05 --- INFO: epoch: 201, batch: 1, loss: 1.379480242729187, modularity_loss: -0.1185215562582016, purity_loss: 1.4074668884277344, regularization_loss: 0.09053495526313782
16:10:05 --- INFO: epoch: 202, batch: 1, loss: 1.3791375160217285, modularity_loss: -0.11851931363344193, purity_loss: 1.4071333408355713, regularization_loss: 0.09052345156669617
16:10:05 --- INFO: epoch: 203, batch: 1, loss: 1.3787963390350342, modularity_loss: -0.11851491779088974, purity_loss: 1.4067989587783813, regularization_loss: 0.09051232039928436
16:10:05 --- INFO: epoch: 204, batch: 1, loss: 1.3784538507461548, modularity_loss: -0.11850925534963608, purity_loss: 1.4064605236053467, regularization_loss: 0.09050260484218597
16:10:05 --- INFO: epoch: 205, batch: 1, loss: 1.378106713294983, modularity_loss: -0.11850284039974213, purity_loss: 1.4061146974563599, regularization_loss: 0.09049486368894577
16:10:05 --- INFO: epoch: 206, batch: 1, loss: 1.3777507543563843, modularity_loss: -0.1184956505894661, purity_loss: 1.4057573080062866, regularization_loss: 0.09048906713724136
16:10:05 --- INFO: epoch: 207, batch: 1, loss: 1.3773839473724365, modularity_loss: -0.11848726123571396, purity_loss: 1.4053865671157837, regularization_loss: 0.09048459678888321
16:10:05 --- INFO: epoch: 208, batch: 1, loss: 1.377004861831665, modularity_loss: -0.11847683042287827, purity_loss: 1.4050012826919556, regularization_loss: 0.09048045426607132
16:10:05 --- INFO: epoch: 209, batch: 1, loss: 1.376611590385437, modularity_loss: -0.11846312135457993, purity_loss: 1.4045994281768799, regularization_loss: 0.09047526121139526
16:10:05 --- INFO: epoch: 210, batch: 1, loss: 1.376198649406433, modularity_loss: -0.11844462901353836, purity_loss: 1.4041756391525269, regularization_loss: 0.0904676541686058
16:10:05 --- INFO: epoch: 211, batch: 1, loss: 1.3757609128952026, modularity_loss: -0.11841986328363419, purity_loss: 1.4037245512008667, regularization_loss: 0.09045626223087311
16:10:05 --- INFO: epoch: 212, batch: 1, loss: 1.3752907514572144, modularity_loss: -0.11838709563016891, purity_loss: 1.4032377004623413, regularization_loss: 0.09044016897678375
16:10:05 --- INFO: epoch: 213, batch: 1, loss: 1.374778389930725, modularity_loss: -0.118344746530056, purity_loss: 1.4027045965194702, regularization_loss: 0.09041856229305267
16:10:05 --- INFO: epoch: 214, batch: 1, loss: 1.3742116689682007, modularity_loss: -0.11829108744859695, purity_loss: 1.4021117687225342, regularization_loss: 0.09039095789194107
16:10:05 --- INFO: epoch: 215, batch: 1, loss: 1.373573899269104, modularity_loss: -0.11822409182786942, purity_loss: 1.401440978050232, regularization_loss: 0.0903569757938385
16:10:05 --- INFO: epoch: 216, batch: 1, loss: 1.3728430271148682, modularity_loss: -0.11814139783382416, purity_loss: 1.4006683826446533, regularization_loss: 0.09031607210636139
16:10:05 --- INFO: epoch: 217, batch: 1, loss: 1.3719875812530518, modularity_loss: -0.11803943663835526, purity_loss: 1.3997600078582764, regularization_loss: 0.09026705473661423
16:10:05 --- INFO: epoch: 218, batch: 1, loss: 1.3709622621536255, modularity_loss: -0.11791301518678665, purity_loss: 1.398667335510254, regularization_loss: 0.09020793437957764
16:10:05 --- INFO: epoch: 219, batch: 1, loss: 1.3697000741958618, modularity_loss: -0.11775419861078262, purity_loss: 1.3973190784454346, regularization_loss: 0.09013521671295166
16:10:05 --- INFO: epoch: 220, batch: 1, loss: 1.3681011199951172, modularity_loss: -0.11755061149597168, purity_loss: 1.3956083059310913, regularization_loss: 0.09004337340593338
16:10:05 --- INFO: epoch: 221, batch: 1, loss: 1.3660086393356323, modularity_loss: -0.11728306114673615, purity_loss: 1.3933680057525635, regularization_loss: 0.08992373198270798
16:10:05 --- INFO: epoch: 222, batch: 1, loss: 1.3631739616394043, modularity_loss: -0.11692159622907639, purity_loss: 1.390332818031311, regularization_loss: 0.08976275473833084
16:10:05 --- INFO: epoch: 223, batch: 1, loss: 1.3591926097869873, modularity_loss: -0.11641881614923477, purity_loss: 1.3860723972320557, regularization_loss: 0.08953902125358582
16:10:05 --- INFO: epoch: 224, batch: 1, loss: 1.3534003496170044, modularity_loss: -0.11570122838020325, purity_loss: 1.3798822164535522, regularization_loss: 0.08921937644481659
16:10:05 --- INFO: epoch: 225, batch: 1, loss: 1.3447415828704834, modularity_loss: -0.1146613135933876, purity_loss: 1.3706485033035278, regularization_loss: 0.08875442296266556
16:10:05 --- INFO: epoch: 226, batch: 1, loss: 1.3317468166351318, modularity_loss: -0.1131705567240715, purity_loss: 1.3568347692489624, regularization_loss: 0.08808265626430511
16:10:05 --- INFO: epoch: 227, batch: 1, loss: 1.3131537437438965, modularity_loss: -0.11116888374090195, purity_loss: 1.3371552228927612, regularization_loss: 0.08716742694377899
16:10:05 --- INFO: epoch: 228, batch: 1, loss: 1.2901880741119385, modularity_loss: -0.10888510942459106, purity_loss: 1.3129775524139404, regularization_loss: 0.08609561622142792
16:10:05 --- INFO: epoch: 229, batch: 1, loss: 1.2699755430221558, modularity_loss: -0.10700159519910812, purity_loss: 1.291811466217041, regularization_loss: 0.08516568690538406
16:10:05 --- INFO: epoch: 230, batch: 1, loss: 1.2629457712173462, modularity_loss: -0.10619644820690155, purity_loss: 1.2844475507736206, regularization_loss: 0.08469466120004654
16:10:05 --- INFO: epoch: 231, batch: 1, loss: 1.2659660577774048, modularity_loss: -0.10611327737569809, purity_loss: 1.2876026630401611, regularization_loss: 0.08447670191526413
16:10:05 --- INFO: epoch: 232, batch: 1, loss: 1.2598941326141357, modularity_loss: -0.10568933188915253, purity_loss: 1.2817015647888184, regularization_loss: 0.08388186246156693
16:10:05 --- INFO: epoch: 233, batch: 1, loss: 1.2402704954147339, modularity_loss: -0.10468912869691849, purity_loss: 1.2621833086013794, regularization_loss: 0.08277627825737
16:10:05 --- INFO: epoch: 234, batch: 1, loss: 1.218919277191162, modularity_loss: -0.10354068130254745, purity_loss: 1.24094820022583, regularization_loss: 0.08151178807020187
16:10:05 --- INFO: epoch: 235, batch: 1, loss: 1.2059345245361328, modularity_loss: -0.10262446850538254, purity_loss: 1.2281144857406616, regularization_loss: 0.0804445669054985
16:10:05 --- INFO: epoch: 236, batch: 1, loss: 1.1994549036026, modularity_loss: -0.1019492968916893, purity_loss: 1.2217668294906616, regularization_loss: 0.07963742315769196
16:10:05 --- INFO: epoch: 237, batch: 1, loss: 1.1915595531463623, modularity_loss: -0.10133221745491028, purity_loss: 1.2139352560043335, regularization_loss: 0.07895646244287491
16:10:05 --- INFO: epoch: 238, batch: 1, loss: 1.1786296367645264, modularity_loss: -0.1007009819149971, purity_loss: 1.2010014057159424, regularization_loss: 0.07832924276590347
16:10:05 --- INFO: epoch: 239, batch: 1, loss: 1.162376046180725, modularity_loss: -0.10010426491498947, purity_loss: 1.1846848726272583, regularization_loss: 0.07779548317193985
16:10:05 --- INFO: epoch: 240, batch: 1, loss: 1.1463439464569092, modularity_loss: -0.09960810840129852, purity_loss: 1.1685312986373901, regularization_loss: 0.07742080837488174
16:10:05 --- INFO: epoch: 241, batch: 1, loss: 1.1332695484161377, modularity_loss: -0.09921819716691971, purity_loss: 1.1552703380584717, regularization_loss: 0.0772174671292305
16:10:05 --- INFO: epoch: 242, batch: 1, loss: 1.1235905885696411, modularity_loss: -0.09889195114374161, purity_loss: 1.1453620195388794, regularization_loss: 0.07712056487798691
16:10:05 --- INFO: epoch: 243, batch: 1, loss: 1.1153900623321533, modularity_loss: -0.09860440343618393, purity_loss: 1.1369625329971313, regularization_loss: 0.07703187316656113
16:10:05 --- INFO: epoch: 244, batch: 1, loss: 1.1061975955963135, modularity_loss: -0.09839081764221191, purity_loss: 1.127693772315979, regularization_loss: 0.07689468562602997
16:10:05 --- INFO: epoch: 245, batch: 1, loss: 1.09506094455719, modularity_loss: -0.09830658882856369, purity_loss: 1.1166490316390991, regularization_loss: 0.07671855390071869
16:10:05 --- INFO: epoch: 246, batch: 1, loss: 1.082991123199463, modularity_loss: -0.09839197993278503, purity_loss: 1.1048362255096436, regularization_loss: 0.07654689997434616
16:10:05 --- INFO: epoch: 247, batch: 1, loss: 1.072371482849121, modularity_loss: -0.09861160814762115, purity_loss: 1.0945744514465332, regularization_loss: 0.07640863955020905
16:10:05 --- INFO: epoch: 248, batch: 1, loss: 1.0644826889038086, modularity_loss: -0.09890683740377426, purity_loss: 1.0870779752731323, regularization_loss: 0.07631158828735352
16:10:05 --- INFO: epoch: 249, batch: 1, loss: 1.0587934255599976, modularity_loss: -0.09918542951345444, purity_loss: 1.0817338228225708, regularization_loss: 0.07624505460262299
16:10:05 --- INFO: epoch: 250, batch: 1, loss: 1.0534627437591553, modularity_loss: -0.09941042214632034, purity_loss: 1.0766702890396118, regularization_loss: 0.07620285451412201
16:10:05 --- INFO: epoch: 251, batch: 1, loss: 1.0472749471664429, modularity_loss: -0.09957806766033173, purity_loss: 1.0706692934036255, regularization_loss: 0.07618370652198792
16:10:05 --- INFO: epoch: 252, batch: 1, loss: 1.0403308868408203, modularity_loss: -0.09970454126596451, purity_loss: 1.0638444423675537, regularization_loss: 0.07619092613458633
16:10:05 --- INFO: epoch: 253, batch: 1, loss: 1.0337318181991577, modularity_loss: -0.09981370717287064, purity_loss: 1.0573208332061768, regularization_loss: 0.07622472196817398
16:10:05 --- INFO: epoch: 254, batch: 1, loss: 1.0283457040786743, modularity_loss: -0.09992269426584244, purity_loss: 1.0519939661026, regularization_loss: 0.07627443224191666
16:10:06 --- INFO: epoch: 255, batch: 1, loss: 1.0240553617477417, modularity_loss: -0.10004276037216187, purity_loss: 1.0477800369262695, regularization_loss: 0.07631805539131165
16:10:06 --- INFO: epoch: 256, batch: 1, loss: 1.0199041366577148, modularity_loss: -0.1001775711774826, purity_loss: 1.0437499284744263, regularization_loss: 0.07633182406425476
16:10:06 --- INFO: epoch: 257, batch: 1, loss: 1.015187382698059, modularity_loss: -0.10032325237989426, purity_loss: 1.0392065048217773, regularization_loss: 0.0763041079044342
16:10:06 --- INFO: epoch: 258, batch: 1, loss: 1.0102074146270752, modularity_loss: -0.10047050565481186, purity_loss: 1.0344361066818237, regularization_loss: 0.07624184340238571
16:10:06 --- INFO: epoch: 259, batch: 1, loss: 1.0057473182678223, modularity_loss: -0.10060497373342514, purity_loss: 1.0301868915557861, regularization_loss: 0.07616536319255829
16:10:06 --- INFO: epoch: 260, batch: 1, loss: 1.0021477937698364, modularity_loss: -0.10071676224470139, purity_loss: 1.0267678499221802, regularization_loss: 0.07609675079584122
16:10:06 --- INFO: epoch: 261, batch: 1, loss: 0.9991469979286194, modularity_loss: -0.10080548375844955, purity_loss: 1.0239002704620361, regularization_loss: 0.07605219632387161
16:10:06 --- INFO: epoch: 262, batch: 1, loss: 0.9962390661239624, modularity_loss: -0.10087530314922333, purity_loss: 1.021075963973999, regularization_loss: 0.07603839039802551
16:10:06 --- INFO: epoch: 263, batch: 1, loss: 0.9932284355163574, modularity_loss: -0.10092931240797043, purity_loss: 1.0181041955947876, regularization_loss: 0.07605355978012085
16:10:06 --- INFO: epoch: 264, batch: 1, loss: 0.9902776479721069, modularity_loss: -0.10097794979810715, purity_loss: 1.0151654481887817, regularization_loss: 0.07609012722969055
16:10:06 --- INFO: epoch: 265, batch: 1, loss: 0.987578809261322, modularity_loss: -0.10103337466716766, purity_loss: 1.012475609779358, regularization_loss: 0.07613659650087357
16:10:06 --- INFO: epoch: 266, batch: 1, loss: 0.9851868152618408, modularity_loss: -0.10110508650541306, purity_loss: 1.0101112127304077, regularization_loss: 0.07618068903684616
16:10:06 --- INFO: epoch: 267, batch: 1, loss: 0.9830668568611145, modularity_loss: -0.10119788348674774, purity_loss: 1.0080515146255493, regularization_loss: 0.07621322572231293
16:10:06 --- INFO: epoch: 268, batch: 1, loss: 0.9811410903930664, modularity_loss: -0.10130982100963593, purity_loss: 1.0062214136123657, regularization_loss: 0.07622946798801422
16:10:06 --- INFO: epoch: 269, batch: 1, loss: 0.9793511033058167, modularity_loss: -0.10143285244703293, purity_loss: 1.004554271697998, regularization_loss: 0.07622967660427094
16:10:06 --- INFO: epoch: 270, batch: 1, loss: 0.977660059928894, modularity_loss: -0.10155776143074036, purity_loss: 1.0029996633529663, regularization_loss: 0.07621815800666809
16:10:06 --- INFO: epoch: 271, batch: 1, loss: 0.9760362505912781, modularity_loss: -0.1016775444149971, purity_loss: 1.001511812210083, regularization_loss: 0.07620197534561157
16:10:06 --- INFO: epoch: 272, batch: 1, loss: 0.9744961857795715, modularity_loss: -0.10178899765014648, purity_loss: 1.0000964403152466, regularization_loss: 0.07618872076272964
16:10:06 --- INFO: epoch: 273, batch: 1, loss: 0.9731033444404602, modularity_loss: -0.10188832879066467, purity_loss: 0.9988073706626892, regularization_loss: 0.07618432492017746
16:10:06 --- INFO: epoch: 274, batch: 1, loss: 0.97188401222229, modularity_loss: -0.10196997225284576, purity_loss: 0.9976638555526733, regularization_loss: 0.07619015872478485
16:10:06 --- INFO: epoch: 275, batch: 1, loss: 0.9707903861999512, modularity_loss: -0.10203143954277039, purity_loss: 0.9966190457344055, regularization_loss: 0.07620278000831604
16:10:06 --- INFO: epoch: 276, batch: 1, loss: 0.9697246551513672, modularity_loss: -0.10207188874483109, purity_loss: 0.9955796599388123, regularization_loss: 0.07621688395738602
16:10:06 --- INFO: epoch: 277, batch: 1, loss: 0.9686285257339478, modularity_loss: -0.10209421068429947, purity_loss: 0.9944941401481628, regularization_loss: 0.076228566467762
16:10:06 --- INFO: epoch: 278, batch: 1, loss: 0.9675381183624268, modularity_loss: -0.1021035686135292, purity_loss: 0.9934043884277344, regularization_loss: 0.07623729854822159
16:10:06 --- INFO: epoch: 279, batch: 1, loss: 0.9665464758872986, modularity_loss: -0.10210707783699036, purity_loss: 0.9924087524414062, regularization_loss: 0.0762447938323021
16:10:06 --- INFO: epoch: 280, batch: 1, loss: 0.9656941294670105, modularity_loss: -0.10211168229579926, purity_loss: 0.9915530681610107, regularization_loss: 0.0762527659535408
16:10:06 --- INFO: epoch: 281, batch: 1, loss: 0.9649245738983154, modularity_loss: -0.10212356597185135, purity_loss: 0.9907869696617126, regularization_loss: 0.07626114785671234
16:10:06 --- INFO: epoch: 282, batch: 1, loss: 0.9641575813293457, modularity_loss: -0.10214591026306152, purity_loss: 0.9900347590446472, regularization_loss: 0.0762687399983406
16:10:06 --- INFO: epoch: 283, batch: 1, loss: 0.9633617997169495, modularity_loss: -0.10217882692813873, purity_loss: 0.9892658591270447, regularization_loss: 0.07627477496862411
16:10:06 --- INFO: epoch: 284, batch: 1, loss: 0.9625717997550964, modularity_loss: -0.10222025960683823, purity_loss: 0.9885129928588867, regularization_loss: 0.07627905160188675
16:10:06 --- INFO: epoch: 285, batch: 1, loss: 0.9618406295776367, modularity_loss: -0.10226700454950333, purity_loss: 0.9878262281417847, regularization_loss: 0.07628142833709717
16:10:06 --- INFO: epoch: 286, batch: 1, loss: 0.9611777663230896, modularity_loss: -0.10231400281190872, purity_loss: 0.9872100949287415, regularization_loss: 0.07628167420625687
16:10:06 --- INFO: epoch: 287, batch: 1, loss: 0.9605622291564941, modularity_loss: -0.10235803574323654, purity_loss: 0.9866412281990051, regularization_loss: 0.07627905905246735
16:10:06 --- INFO: epoch: 288, batch: 1, loss: 0.9599660634994507, modularity_loss: -0.1023944690823555, purity_loss: 0.9860861897468567, regularization_loss: 0.0762743204832077
16:10:06 --- INFO: epoch: 289, batch: 1, loss: 0.9593716263771057, modularity_loss: -0.10242114216089249, purity_loss: 0.985523521900177, regularization_loss: 0.0762692540884018
16:10:06 --- INFO: epoch: 290, batch: 1, loss: 0.9587939977645874, modularity_loss: -0.10243899375200272, purity_loss: 0.9849668741226196, regularization_loss: 0.0762660875916481
16:10:06 --- INFO: epoch: 291, batch: 1, loss: 0.9582525491714478, modularity_loss: -0.10244935750961304, purity_loss: 0.9844349026679993, regularization_loss: 0.07626698911190033
16:10:06 --- INFO: epoch: 292, batch: 1, loss: 0.9577414989471436, modularity_loss: -0.10245458781719208, purity_loss: 0.9839234352111816, regularization_loss: 0.0762726366519928
16:10:06 --- INFO: epoch: 293, batch: 1, loss: 0.9572569131851196, modularity_loss: -0.10245797783136368, purity_loss: 0.9834330081939697, regularization_loss: 0.07628185302019119
16:10:06 --- INFO: epoch: 294, batch: 1, loss: 0.9567923545837402, modularity_loss: -0.10246121138334274, purity_loss: 0.9829609990119934, regularization_loss: 0.07629258930683136
16:10:06 --- INFO: epoch: 295, batch: 1, loss: 0.9563460946083069, modularity_loss: -0.10246530920267105, purity_loss: 0.9825089573860168, regularization_loss: 0.07630244642496109
16:10:06 --- INFO: epoch: 296, batch: 1, loss: 0.9559183716773987, modularity_loss: -0.10247078537940979, purity_loss: 0.9820798635482788, regularization_loss: 0.07630927115678787
16:10:06 --- INFO: epoch: 297, batch: 1, loss: 0.9555025100708008, modularity_loss: -0.10247761011123657, purity_loss: 0.9816682934761047, regularization_loss: 0.07631184905767441
16:10:06 --- INFO: epoch: 298, batch: 1, loss: 0.9550981521606445, modularity_loss: -0.10248620063066483, purity_loss: 0.9812746644020081, regularization_loss: 0.0763097032904625
16:10:06 --- INFO: epoch: 299, batch: 1, loss: 0.9547106027603149, modularity_loss: -0.10249633342027664, purity_loss: 0.9809028506278992, regularization_loss: 0.07630406320095062
16:10:06 --- INFO: epoch: 300, batch: 1, loss: 0.9543471336364746, modularity_loss: -0.10250724852085114, purity_loss: 0.9805569648742676, regularization_loss: 0.07629740238189697
16:10:06 --- INFO: epoch: 301, batch: 1, loss: 0.9539975523948669, modularity_loss: -0.10251773148775101, purity_loss: 0.9802230000495911, regularization_loss: 0.07629226893186569
16:10:06 --- INFO: epoch: 302, batch: 1, loss: 0.9536591172218323, modularity_loss: -0.1025269404053688, purity_loss: 0.9798956513404846, regularization_loss: 0.07629039883613586
16:10:06 --- INFO: epoch: 303, batch: 1, loss: 0.9533283710479736, modularity_loss: -0.1025349423289299, purity_loss: 0.9795710444450378, regularization_loss: 0.0762922391295433
16:10:06 --- INFO: epoch: 304, batch: 1, loss: 0.9530110359191895, modularity_loss: -0.10254176706075668, purity_loss: 0.9792556762695312, regularization_loss: 0.07629713416099548
16:10:06 --- INFO: epoch: 305, batch: 1, loss: 0.9527117013931274, modularity_loss: -0.10254742950201035, purity_loss: 0.9789555668830872, regularization_loss: 0.07630358636379242
16:10:06 --- INFO: epoch: 306, batch: 1, loss: 0.9524286985397339, modularity_loss: -0.10255195200443268, purity_loss: 0.9786710143089294, regularization_loss: 0.07630962878465652
16:10:06 --- INFO: epoch: 307, batch: 1, loss: 0.9521592855453491, modularity_loss: -0.1025555282831192, purity_loss: 0.9784007668495178, regularization_loss: 0.07631406933069229
16:10:06 --- INFO: epoch: 308, batch: 1, loss: 0.9518950581550598, modularity_loss: -0.10255823284387589, purity_loss: 0.9781373143196106, regularization_loss: 0.0763159915804863
16:10:06 --- INFO: epoch: 309, batch: 1, loss: 0.9516329169273376, modularity_loss: -0.10256015509366989, purity_loss: 0.9778774976730347, regularization_loss: 0.07631558924913406
16:10:06 --- INFO: epoch: 310, batch: 1, loss: 0.9513803720474243, modularity_loss: -0.10256243497133255, purity_loss: 0.9776294231414795, regularization_loss: 0.07631341367959976
16:10:06 --- INFO: epoch: 311, batch: 1, loss: 0.951138973236084, modularity_loss: -0.10256528109312057, purity_loss: 0.977393388748169, regularization_loss: 0.07631086558103561
16:10:06 --- INFO: epoch: 312, batch: 1, loss: 0.9509060978889465, modularity_loss: -0.10256869345903397, purity_loss: 0.9771656394004822, regularization_loss: 0.07630913704633713
16:10:06 --- INFO: epoch: 313, batch: 1, loss: 0.9506793022155762, modularity_loss: -0.10257240384817123, purity_loss: 0.9769428968429565, regularization_loss: 0.07630879431962967
16:10:06 --- INFO: epoch: 314, batch: 1, loss: 0.9504570364952087, modularity_loss: -0.10257606953382492, purity_loss: 0.9767231941223145, regularization_loss: 0.07630990445613861
16:10:06 --- INFO: epoch: 315, batch: 1, loss: 0.9502395987510681, modularity_loss: -0.10257917642593384, purity_loss: 0.9765068292617798, regularization_loss: 0.07631192356348038
16:10:06 --- INFO: epoch: 316, batch: 1, loss: 0.9500161409378052, modularity_loss: -0.10258106887340546, purity_loss: 0.9762831926345825, regularization_loss: 0.0763140469789505
16:10:06 --- INFO: epoch: 317, batch: 1, loss: 0.9497948288917542, modularity_loss: -0.10258343815803528, purity_loss: 0.9760638475418091, regularization_loss: 0.07631439715623856
16:10:06 --- INFO: epoch: 318, batch: 1, loss: 0.9495781064033508, modularity_loss: -0.10258594900369644, purity_loss: 0.9758509993553162, regularization_loss: 0.07631304115056992
16:10:06 --- INFO: epoch: 319, batch: 1, loss: 0.9493682384490967, modularity_loss: -0.10258853435516357, purity_loss: 0.9756460785865784, regularization_loss: 0.07631072402000427
16:10:06 --- INFO: epoch: 320, batch: 1, loss: 0.9491637349128723, modularity_loss: -0.10259110480546951, purity_loss: 0.9754467010498047, regularization_loss: 0.07630815356969833
16:10:06 --- INFO: epoch: 321, batch: 1, loss: 0.9489652514457703, modularity_loss: -0.10259312391281128, purity_loss: 0.9752519726753235, regularization_loss: 0.07630641013383865
16:10:06 --- INFO: epoch: 322, batch: 1, loss: 0.9487718939781189, modularity_loss: -0.10259420424699783, purity_loss: 0.9750600457191467, regularization_loss: 0.0763060599565506
16:10:06 --- INFO: epoch: 323, batch: 1, loss: 0.9485825896263123, modularity_loss: -0.10259406268596649, purity_loss: 0.9748694896697998, regularization_loss: 0.07630714774131775
16:10:06 --- INFO: epoch: 324, batch: 1, loss: 0.9483983516693115, modularity_loss: -0.10259247571229935, purity_loss: 0.9746814370155334, regularization_loss: 0.07630936801433563
16:10:06 --- INFO: epoch: 325, batch: 1, loss: 0.9482191801071167, modularity_loss: -0.10258950293064117, purity_loss: 0.974496603012085, regularization_loss: 0.07631208747625351
16:10:06 --- INFO: epoch: 326, batch: 1, loss: 0.9480444192886353, modularity_loss: -0.10258518159389496, purity_loss: 0.9743149876594543, regularization_loss: 0.07631459087133408
16:10:06 --- INFO: epoch: 327, batch: 1, loss: 0.9478735327720642, modularity_loss: -0.10257986932992935, purity_loss: 0.9741369485855103, regularization_loss: 0.0763164609670639
16:10:06 --- INFO: epoch: 328, batch: 1, loss: 0.947706401348114, modularity_loss: -0.10257402807474136, purity_loss: 0.9739629626274109, regularization_loss: 0.0763174444437027
16:10:06 --- INFO: epoch: 329, batch: 1, loss: 0.9475415945053101, modularity_loss: -0.10256823897361755, purity_loss: 0.9737921953201294, regularization_loss: 0.07631760835647583
16:10:06 --- INFO: epoch: 330, batch: 1, loss: 0.947380006313324, modularity_loss: -0.10256295651197433, purity_loss: 0.973625898361206, regularization_loss: 0.07631705701351166
16:10:06 --- INFO: epoch: 331, batch: 1, loss: 0.947222888469696, modularity_loss: -0.10255860537290573, purity_loss: 0.9734653234481812, regularization_loss: 0.07631615549325943
16:10:06 --- INFO: epoch: 332, batch: 1, loss: 0.9470693469047546, modularity_loss: -0.10255543142557144, purity_loss: 0.9733095169067383, regularization_loss: 0.0763152539730072
16:10:06 --- INFO: epoch: 333, batch: 1, loss: 0.9469189047813416, modularity_loss: -0.10255343466997147, purity_loss: 0.9731577634811401, regularization_loss: 0.07631459087133408
16:10:06 --- INFO: epoch: 334, batch: 1, loss: 0.9467717409133911, modularity_loss: -0.10255228728055954, purity_loss: 0.9730098247528076, regularization_loss: 0.07631419599056244
16:10:06 --- INFO: epoch: 335, batch: 1, loss: 0.9466280341148376, modularity_loss: -0.10255156457424164, purity_loss: 0.9728655219078064, regularization_loss: 0.07631409913301468
16:10:06 --- INFO: epoch: 336, batch: 1, loss: 0.94648677110672, modularity_loss: -0.10255078971385956, purity_loss: 0.9727234244346619, regularization_loss: 0.07631411403417587
16:10:06 --- INFO: epoch: 337, batch: 1, loss: 0.9463483095169067, modularity_loss: -0.10254957526922226, purity_loss: 0.9725837111473083, regularization_loss: 0.07631418108940125
16:10:06 --- INFO: epoch: 338, batch: 1, loss: 0.9462122321128845, modularity_loss: -0.10254774242639542, purity_loss: 0.9724457263946533, regularization_loss: 0.07631426304578781
16:10:06 --- INFO: epoch: 339, batch: 1, loss: 0.9460793733596802, modularity_loss: -0.10254526138305664, purity_loss: 0.9723101854324341, regularization_loss: 0.07631447166204453
16:10:06 --- INFO: epoch: 340, batch: 1, loss: 0.9459496736526489, modularity_loss: -0.10254238545894623, purity_loss: 0.9721770882606506, regularization_loss: 0.07631496340036392
16:10:06 --- INFO: epoch: 341, batch: 1, loss: 0.945822536945343, modularity_loss: -0.10253932327032089, purity_loss: 0.9720460772514343, regularization_loss: 0.07631578296422958
16:10:06 --- INFO: epoch: 342, batch: 1, loss: 0.9456979632377625, modularity_loss: -0.10253629833459854, purity_loss: 0.971917450428009, regularization_loss: 0.07631683349609375
16:10:06 --- INFO: epoch: 343, batch: 1, loss: 0.9455756545066833, modularity_loss: -0.10253339260816574, purity_loss: 0.97179114818573, regularization_loss: 0.07631789892911911
16:10:06 --- INFO: epoch: 344, batch: 1, loss: 0.9454553127288818, modularity_loss: -0.10253065079450607, purity_loss: 0.9716672897338867, regularization_loss: 0.07631870359182358
16:10:06 --- INFO: epoch: 345, batch: 1, loss: 0.9453374147415161, modularity_loss: -0.10252805054187775, purity_loss: 0.9715464115142822, regularization_loss: 0.07631904631853104
16:10:06 --- INFO: epoch: 346, batch: 1, loss: 0.9452214241027832, modularity_loss: -0.1025255098938942, purity_loss: 0.9714280962944031, regularization_loss: 0.07631884515285492
16:10:06 --- INFO: epoch: 347, batch: 1, loss: 0.9451086521148682, modularity_loss: -0.10252300649881363, purity_loss: 0.9713133573532104, regularization_loss: 0.07631832361221313
16:10:06 --- INFO: epoch: 348, batch: 1, loss: 0.9449979066848755, modularity_loss: -0.10252051800489426, purity_loss: 0.9712008237838745, regularization_loss: 0.07631760835647583
16:10:06 --- INFO: epoch: 349, batch: 1, loss: 0.9448889493942261, modularity_loss: -0.10251802206039429, purity_loss: 0.9710900187492371, regularization_loss: 0.0763169527053833
16:10:06 --- INFO: epoch: 350, batch: 1, loss: 0.9447805285453796, modularity_loss: -0.10251545161008835, purity_loss: 0.9709794521331787, regularization_loss: 0.07631655037403107
16:10:06 --- INFO: epoch: 351, batch: 1, loss: 0.9446732997894287, modularity_loss: -0.10251347720623016, purity_loss: 0.9708707332611084, regularization_loss: 0.07631605863571167
16:10:06 --- INFO: epoch: 352, batch: 1, loss: 0.9445672035217285, modularity_loss: -0.10251197963953018, purity_loss: 0.9707635045051575, regularization_loss: 0.07631566375494003
16:10:06 --- INFO: epoch: 353, batch: 1, loss: 0.9444625973701477, modularity_loss: -0.10251083225011826, purity_loss: 0.9706579446792603, regularization_loss: 0.0763154849410057
16:10:06 --- INFO: epoch: 354, batch: 1, loss: 0.944359302520752, modularity_loss: -0.10250979661941528, purity_loss: 0.9705535769462585, regularization_loss: 0.0763154998421669
16:10:06 --- INFO: epoch: 355, batch: 1, loss: 0.9442577958106995, modularity_loss: -0.10250864177942276, purity_loss: 0.9704508185386658, regularization_loss: 0.07631563395261765
16:10:06 --- INFO: epoch: 356, batch: 1, loss: 0.9441581964492798, modularity_loss: -0.10250716656446457, purity_loss: 0.9703496098518372, regularization_loss: 0.0763157531619072
16:10:06 --- INFO: epoch: 357, batch: 1, loss: 0.9440597295761108, modularity_loss: -0.10250525921583176, purity_loss: 0.9702492356300354, regularization_loss: 0.07631576061248779
16:10:07 --- INFO: epoch: 358, batch: 1, loss: 0.9439629316329956, modularity_loss: -0.10250284522771835, purity_loss: 0.9701501131057739, regularization_loss: 0.07631563395261765
16:10:07 --- INFO: epoch: 359, batch: 1, loss: 0.943867564201355, modularity_loss: -0.10249997675418854, purity_loss: 0.9700521230697632, regularization_loss: 0.07631540298461914
16:10:07 --- INFO: epoch: 360, batch: 1, loss: 0.9437733292579651, modularity_loss: -0.10249678045511246, purity_loss: 0.9699549674987793, regularization_loss: 0.07631514221429825
16:10:07 --- INFO: epoch: 361, batch: 1, loss: 0.943680465221405, modularity_loss: -0.10249302536249161, purity_loss: 0.9698582291603088, regularization_loss: 0.0763152465224266
16:10:07 --- INFO: epoch: 362, batch: 1, loss: 0.9435888528823853, modularity_loss: -0.1024889275431633, purity_loss: 0.9697620868682861, regularization_loss: 0.07631567865610123
16:10:07 --- INFO: epoch: 363, batch: 1, loss: 0.9434982538223267, modularity_loss: -0.10248468071222305, purity_loss: 0.969666600227356, regularization_loss: 0.07631631940603256
16:10:07 --- INFO: epoch: 364, batch: 1, loss: 0.9434094429016113, modularity_loss: -0.10248053818941116, purity_loss: 0.9695729613304138, regularization_loss: 0.07631698995828629
16:10:07 --- INFO: epoch: 365, batch: 1, loss: 0.943321943283081, modularity_loss: -0.10247695446014404, purity_loss: 0.9694816470146179, regularization_loss: 0.07631723582744598
16:10:07 --- INFO: epoch: 366, batch: 1, loss: 0.9432352185249329, modularity_loss: -0.10247398912906647, purity_loss: 0.9693921804428101, regularization_loss: 0.07631701976060867
16:10:07 --- INFO: epoch: 367, batch: 1, loss: 0.9431496858596802, modularity_loss: -0.10247162729501724, purity_loss: 0.9693048596382141, regularization_loss: 0.07631644606590271
16:10:07 --- INFO: epoch: 368, batch: 1, loss: 0.9430649280548096, modularity_loss: -0.10246987640857697, purity_loss: 0.9692191481590271, regularization_loss: 0.07631565630435944
16:10:07 --- INFO: epoch: 369, batch: 1, loss: 0.9429814219474792, modularity_loss: -0.10246854275465012, purity_loss: 0.9691351056098938, regularization_loss: 0.07631484419107437
16:10:07 --- INFO: epoch: 370, batch: 1, loss: 0.9428983926773071, modularity_loss: -0.10246701538562775, purity_loss: 0.969050943851471, regularization_loss: 0.07631449401378632
16:10:07 --- INFO: epoch: 371, batch: 1, loss: 0.9428155422210693, modularity_loss: -0.10246492177248001, purity_loss: 0.9689658284187317, regularization_loss: 0.07631461322307587
16:10:07 --- INFO: epoch: 372, batch: 1, loss: 0.9427334666252136, modularity_loss: -0.10246232151985168, purity_loss: 0.9688807129859924, regularization_loss: 0.07631506025791168
16:10:07 --- INFO: epoch: 373, batch: 1, loss: 0.9426526427268982, modularity_loss: -0.10245934873819351, purity_loss: 0.9687963724136353, regularization_loss: 0.07631563395261765
16:10:07 --- INFO: epoch: 374, batch: 1, loss: 0.9425736665725708, modularity_loss: -0.10245656222105026, purity_loss: 0.9687142372131348, regularization_loss: 0.07631602138280869
16:10:07 --- INFO: epoch: 375, batch: 1, loss: 0.942495584487915, modularity_loss: -0.10245406627655029, purity_loss: 0.9686335921287537, regularization_loss: 0.07631608843803406
16:10:07 --- INFO: epoch: 376, batch: 1, loss: 0.9424185752868652, modularity_loss: -0.102451853454113, purity_loss: 0.9685546159744263, regularization_loss: 0.07631584256887436
16:10:07 --- INFO: epoch: 377, batch: 1, loss: 0.9423425197601318, modularity_loss: -0.10244999825954437, purity_loss: 0.9684771299362183, regularization_loss: 0.07631538808345795
16:10:07 --- INFO: epoch: 378, batch: 1, loss: 0.9422673583030701, modularity_loss: -0.10244844108819962, purity_loss: 0.9684009552001953, regularization_loss: 0.07631482183933258
16:10:07 --- INFO: epoch: 379, batch: 1, loss: 0.9421930313110352, modularity_loss: -0.10244712233543396, purity_loss: 0.9683258533477783, regularization_loss: 0.076314277946949
16:10:07 --- INFO: epoch: 380, batch: 1, loss: 0.9421196579933167, modularity_loss: -0.10244588553905487, purity_loss: 0.9682517051696777, regularization_loss: 0.07631382346153259
16:10:07 --- INFO: epoch: 381, batch: 1, loss: 0.9420470595359802, modularity_loss: -0.10244457423686981, purity_loss: 0.9681781530380249, regularization_loss: 0.07631345838308334
16:10:07 --- INFO: epoch: 382, batch: 1, loss: 0.9419751763343811, modularity_loss: -0.10244309902191162, purity_loss: 0.968105137348175, regularization_loss: 0.07631315290927887
16:10:07 --- INFO: epoch: 383, batch: 1, loss: 0.9419039487838745, modularity_loss: -0.10244134813547134, purity_loss: 0.9680324792861938, regularization_loss: 0.0763128325343132
16:10:07 --- INFO: epoch: 384, batch: 1, loss: 0.9418334364891052, modularity_loss: -0.10243934392929077, purity_loss: 0.9679602980613708, regularization_loss: 0.07631250470876694
16:10:07 --- INFO: epoch: 385, batch: 1, loss: 0.9417638778686523, modularity_loss: -0.1024370864033699, purity_loss: 0.9678887724876404, regularization_loss: 0.07631216198205948
16:10:07 --- INFO: epoch: 386, batch: 1, loss: 0.9416946768760681, modularity_loss: -0.1024346724152565, purity_loss: 0.9678175449371338, regularization_loss: 0.07631182670593262
16:10:07 --- INFO: epoch: 387, batch: 1, loss: 0.9416260719299316, modularity_loss: -0.10243221372365952, purity_loss: 0.9677467346191406, regularization_loss: 0.07631154358386993
16:10:07 --- INFO: epoch: 388, batch: 1, loss: 0.9415590167045593, modularity_loss: -0.10242979228496552, purity_loss: 0.9676774740219116, regularization_loss: 0.07631134241819382
16:10:07 --- INFO: epoch: 389, batch: 1, loss: 0.9414922595024109, modularity_loss: -0.10242704302072525, purity_loss: 0.9676076769828796, regularization_loss: 0.07631160318851471
16:10:07 --- INFO: epoch: 390, batch: 1, loss: 0.9414258003234863, modularity_loss: -0.10242410004138947, purity_loss: 0.9675377607345581, regularization_loss: 0.07631213963031769
16:10:07 --- INFO: epoch: 391, batch: 1, loss: 0.9413596987724304, modularity_loss: -0.10242173075675964, purity_loss: 0.9674693942070007, regularization_loss: 0.07631202787160873
16:10:07 --- INFO: epoch: 392, batch: 1, loss: 0.941293478012085, modularity_loss: -0.10241984575986862, purity_loss: 0.9674020409584045, regularization_loss: 0.07631129026412964
16:10:07 --- INFO: epoch: 393, batch: 1, loss: 0.9412277936935425, modularity_loss: -0.10241839289665222, purity_loss: 0.967336118221283, regularization_loss: 0.07631006836891174
16:10:07 --- INFO: epoch: 394, batch: 1, loss: 0.9411621689796448, modularity_loss: -0.10241726785898209, purity_loss: 0.9672708511352539, regularization_loss: 0.07630858570337296
16:10:07 --- INFO: epoch: 395, batch: 1, loss: 0.9410971403121948, modularity_loss: -0.10241633653640747, purity_loss: 0.9672063589096069, regularization_loss: 0.07630713284015656
16:10:07 --- INFO: epoch: 396, batch: 1, loss: 0.9410321116447449, modularity_loss: -0.10241535305976868, purity_loss: 0.9671415686607361, regularization_loss: 0.07630588859319687
16:10:07 --- INFO: epoch: 397, batch: 1, loss: 0.940966010093689, modularity_loss: -0.10241450369358063, purity_loss: 0.9670761823654175, regularization_loss: 0.07630430907011032
16:10:07 --- INFO: epoch: 398, batch: 1, loss: 0.9409000873565674, modularity_loss: -0.1024136170744896, purity_loss: 0.9670111536979675, regularization_loss: 0.07630255818367004
16:10:07 --- INFO: epoch: 399, batch: 1, loss: 0.9408342838287354, modularity_loss: -0.10241234302520752, purity_loss: 0.9669458270072937, regularization_loss: 0.07630077749490738
16:10:07 --- INFO: epoch: 400, batch: 1, loss: 0.9407693147659302, modularity_loss: -0.1024104654788971, purity_loss: 0.9668806791305542, regularization_loss: 0.07629913091659546
16:10:07 --- INFO: epoch: 401, batch: 1, loss: 0.9407059550285339, modularity_loss: -0.10240786522626877, purity_loss: 0.9668161273002625, regularization_loss: 0.07629767805337906
16:10:07 --- INFO: epoch: 402, batch: 1, loss: 0.9406442046165466, modularity_loss: -0.10240411758422852, purity_loss: 0.9667513370513916, regularization_loss: 0.07629700750112534
16:10:07 --- INFO: epoch: 403, batch: 1, loss: 0.9405823945999146, modularity_loss: -0.10239925980567932, purity_loss: 0.9666846990585327, regularization_loss: 0.07629695534706116
16:10:07 --- INFO: epoch: 404, batch: 1, loss: 0.9405208826065063, modularity_loss: -0.10239361226558685, purity_loss: 0.9666172862052917, regularization_loss: 0.07629723846912384
16:10:07 --- INFO: epoch: 405, batch: 1, loss: 0.9404600858688354, modularity_loss: -0.10238754004240036, purity_loss: 0.9665499925613403, regularization_loss: 0.07629761844873428
16:10:07 --- INFO: epoch: 406, batch: 1, loss: 0.9404001235961914, modularity_loss: -0.10238140821456909, purity_loss: 0.9664837121963501, regularization_loss: 0.0762978196144104
16:10:07 --- INFO: epoch: 407, batch: 1, loss: 0.9403409361839294, modularity_loss: -0.10237560421228409, purity_loss: 0.9664188623428345, regularization_loss: 0.07629766315221786
16:10:07 --- INFO: epoch: 408, batch: 1, loss: 0.9402820467948914, modularity_loss: -0.10237038880586624, purity_loss: 0.9663553237915039, regularization_loss: 0.0762970894575119
16:10:07 --- INFO: epoch: 409, batch: 1, loss: 0.9402234554290771, modularity_loss: -0.10236598551273346, purity_loss: 0.9662933349609375, regularization_loss: 0.0762961134314537
16:10:07 --- INFO: epoch: 410, batch: 1, loss: 0.9401653409004211, modularity_loss: -0.10236240923404694, purity_loss: 0.96623295545578, regularization_loss: 0.07629478722810745
16:10:07 --- INFO: epoch: 411, batch: 1, loss: 0.9401073455810547, modularity_loss: -0.10235923528671265, purity_loss: 0.9661730527877808, regularization_loss: 0.07629355788230896
16:10:07 --- INFO: epoch: 412, batch: 1, loss: 0.940049409866333, modularity_loss: -0.10235641896724701, purity_loss: 0.9661134481430054, regularization_loss: 0.07629239559173584
16:10:07 --- INFO: epoch: 413, batch: 1, loss: 0.9399918913841248, modularity_loss: -0.10235369950532913, purity_loss: 0.966054379940033, regularization_loss: 0.07629123330116272
16:10:07 --- INFO: epoch: 414, batch: 1, loss: 0.939934492111206, modularity_loss: -0.10235105454921722, purity_loss: 0.9659955501556396, regularization_loss: 0.07628999650478363
16:10:07 --- INFO: epoch: 415, batch: 1, loss: 0.939877450466156, modularity_loss: -0.10234836488962173, purity_loss: 0.9659371376037598, regularization_loss: 0.07628868520259857
16:10:07 --- INFO: epoch: 416, batch: 1, loss: 0.9398202896118164, modularity_loss: -0.10234559327363968, purity_loss: 0.9658785462379456, regularization_loss: 0.07628732919692993
16:10:07 --- INFO: epoch: 417, batch: 1, loss: 0.9397633075714111, modularity_loss: -0.10234273970127106, purity_loss: 0.9658200740814209, regularization_loss: 0.07628600299358368
16:10:07 --- INFO: epoch: 418, batch: 1, loss: 0.9397068023681641, modularity_loss: -0.10233983397483826, purity_loss: 0.9657618999481201, regularization_loss: 0.07628476619720459
16:10:07 --- INFO: epoch: 419, batch: 1, loss: 0.9396510720252991, modularity_loss: -0.1023372933268547, purity_loss: 0.9657050967216492, regularization_loss: 0.07628326117992401
16:10:07 --- INFO: epoch: 420, batch: 1, loss: 0.9395960569381714, modularity_loss: -0.10233458131551743, purity_loss: 0.9656485319137573, regularization_loss: 0.0762820765376091
16:10:07 --- INFO: epoch: 421, batch: 1, loss: 0.9395410418510437, modularity_loss: -0.10233163088560104, purity_loss: 0.965591549873352, regularization_loss: 0.07628113776445389
16:10:07 --- INFO: epoch: 422, batch: 1, loss: 0.9394860863685608, modularity_loss: -0.10232839733362198, purity_loss: 0.9655341506004333, regularization_loss: 0.07628034800291061
16:10:07 --- INFO: epoch: 423, batch: 1, loss: 0.939431369304657, modularity_loss: -0.10232485085725784, purity_loss: 0.9654766917228699, regularization_loss: 0.07627955079078674
16:10:07 --- INFO: epoch: 424, batch: 1, loss: 0.9393767714500427, modularity_loss: -0.10232110321521759, purity_loss: 0.9654192328453064, regularization_loss: 0.07627864181995392
16:10:07 --- INFO: epoch: 425, batch: 1, loss: 0.9393222332000732, modularity_loss: -0.10231726616621017, purity_loss: 0.9653618931770325, regularization_loss: 0.07627761363983154
16:10:07 --- INFO: epoch: 426, batch: 1, loss: 0.9392675757408142, modularity_loss: -0.10231351852416992, purity_loss: 0.9653046131134033, regularization_loss: 0.07627648860216141
16:10:07 --- INFO: epoch: 427, batch: 1, loss: 0.9392129778862, modularity_loss: -0.1023099422454834, purity_loss: 0.965247631072998, regularization_loss: 0.0762752965092659
16:10:07 --- INFO: epoch: 428, batch: 1, loss: 0.9391582608222961, modularity_loss: -0.10230664163827896, purity_loss: 0.9651908278465271, regularization_loss: 0.076274074614048
16:10:07 --- INFO: epoch: 429, batch: 1, loss: 0.9391036629676819, modularity_loss: -0.10230355709791183, purity_loss: 0.9651344418525696, regularization_loss: 0.07627277076244354
16:10:07 --- INFO: epoch: 430, batch: 1, loss: 0.9390494227409363, modularity_loss: -0.10230064392089844, purity_loss: 0.9650787115097046, regularization_loss: 0.07627137750387192
16:10:07 --- INFO: epoch: 431, batch: 1, loss: 0.938994824886322, modularity_loss: -0.1022978276014328, purity_loss: 0.9650228023529053, regularization_loss: 0.07626986503601074
16:10:07 --- INFO: epoch: 432, batch: 1, loss: 0.9389402866363525, modularity_loss: -0.10229498893022537, purity_loss: 0.9649670720100403, regularization_loss: 0.07626822590827942
16:10:07 --- INFO: epoch: 433, batch: 1, loss: 0.9388858079910278, modularity_loss: -0.10229197889566422, purity_loss: 0.9649112820625305, regularization_loss: 0.07626649737358093
16:10:07 --- INFO: epoch: 434, batch: 1, loss: 0.9388314485549927, modularity_loss: -0.10228884220123291, purity_loss: 0.9648555517196655, regularization_loss: 0.07626470923423767
16:10:07 --- INFO: epoch: 435, batch: 1, loss: 0.9387768507003784, modularity_loss: -0.10228552669286728, purity_loss: 0.9647994637489319, regularization_loss: 0.07626292109489441
16:10:07 --- INFO: epoch: 436, batch: 1, loss: 0.9387221932411194, modularity_loss: -0.10228202491998672, purity_loss: 0.9647430777549744, regularization_loss: 0.07626116275787354
16:10:07 --- INFO: epoch: 437, batch: 1, loss: 0.9386676549911499, modularity_loss: -0.10227837413549423, purity_loss: 0.9646865725517273, regularization_loss: 0.07625948637723923
16:10:07 --- INFO: epoch: 438, batch: 1, loss: 0.9386130571365356, modularity_loss: -0.10227451473474503, purity_loss: 0.9646297693252563, regularization_loss: 0.07625782489776611
16:10:07 --- INFO: epoch: 439, batch: 1, loss: 0.9385583400726318, modularity_loss: -0.1022704467177391, purity_loss: 0.9645726680755615, regularization_loss: 0.07625613361597061
16:10:07 --- INFO: epoch: 440, batch: 1, loss: 0.9385033249855042, modularity_loss: -0.10226613283157349, purity_loss: 0.9645150899887085, regularization_loss: 0.07625436037778854
16:10:07 --- INFO: epoch: 441, batch: 1, loss: 0.9384485483169556, modularity_loss: -0.10226156562566757, purity_loss: 0.9644576907157898, regularization_loss: 0.07625244557857513
16:10:07 --- INFO: epoch: 442, batch: 1, loss: 0.9383935928344727, modularity_loss: -0.10225679725408554, purity_loss: 0.9643999934196472, regularization_loss: 0.07625041157007217
16:10:07 --- INFO: epoch: 443, batch: 1, loss: 0.9383385181427002, modularity_loss: -0.10225187242031097, purity_loss: 0.9643420577049255, regularization_loss: 0.07624830305576324
16:10:07 --- INFO: epoch: 444, batch: 1, loss: 0.9382832646369934, modularity_loss: -0.10224682092666626, purity_loss: 0.9642839431762695, regularization_loss: 0.07624613493680954
16:10:07 --- INFO: epoch: 445, batch: 1, loss: 0.9382277131080627, modularity_loss: -0.1022416427731514, purity_loss: 0.9642254114151001, regularization_loss: 0.07624392211437225
16:10:07 --- INFO: epoch: 446, batch: 1, loss: 0.9381720423698425, modularity_loss: -0.10223636776208878, purity_loss: 0.9641667604446411, regularization_loss: 0.07624167203903198
16:10:07 --- INFO: epoch: 447, batch: 1, loss: 0.9381164908409119, modularity_loss: -0.1022309735417366, purity_loss: 0.9641081094741821, regularization_loss: 0.07623934000730515
16:10:07 --- INFO: epoch: 448, batch: 1, loss: 0.938060462474823, modularity_loss: -0.1022254005074501, purity_loss: 0.9640489816665649, regularization_loss: 0.07623686641454697
16:10:07 --- INFO: epoch: 449, batch: 1, loss: 0.9380040764808655, modularity_loss: -0.10221963375806808, purity_loss: 0.9639894366264343, regularization_loss: 0.07623425871133804
16:10:07 --- INFO: epoch: 450, batch: 1, loss: 0.9379478693008423, modularity_loss: -0.10221365094184875, purity_loss: 0.9639300107955933, regularization_loss: 0.07623150199651718
16:10:07 --- INFO: epoch: 451, batch: 1, loss: 0.9378921985626221, modularity_loss: -0.10220779478549957, purity_loss: 0.9638718366622925, regularization_loss: 0.07622815668582916
16:10:07 --- INFO: epoch: 452, batch: 1, loss: 0.9378358125686646, modularity_loss: -0.1022019311785698, purity_loss: 0.9638133645057678, regularization_loss: 0.07622438669204712
16:10:07 --- INFO: epoch: 453, batch: 1, loss: 0.9377795457839966, modularity_loss: -0.10219591110944748, purity_loss: 0.9637550711631775, regularization_loss: 0.07622037082910538
16:10:07 --- INFO: epoch: 454, batch: 1, loss: 0.9377227425575256, modularity_loss: -0.10218960046768188, purity_loss: 0.9636960625648499, regularization_loss: 0.07621628046035767
16:10:07 --- INFO: epoch: 455, batch: 1, loss: 0.9376655220985413, modularity_loss: -0.1021827906370163, purity_loss: 0.9636361002922058, regularization_loss: 0.07621221989393234
16:10:07 --- INFO: epoch: 456, batch: 1, loss: 0.9376082420349121, modularity_loss: -0.10217532515525818, purity_loss: 0.9635753631591797, regularization_loss: 0.07620822638273239
16:10:07 --- INFO: epoch: 457, batch: 1, loss: 0.9375503659248352, modularity_loss: -0.10216706246137619, purity_loss: 0.9635131359100342, regularization_loss: 0.07620427757501602
16:10:07 --- INFO: epoch: 458, batch: 1, loss: 0.937491774559021, modularity_loss: -0.10215798765420914, purity_loss: 0.9634494781494141, regularization_loss: 0.07620031386613846
16:10:07 --- INFO: epoch: 459, batch: 1, loss: 0.9374328255653381, modularity_loss: -0.10214810073375702, purity_loss: 0.9633846282958984, regularization_loss: 0.07619631290435791
16:10:07 --- INFO: epoch: 460, batch: 1, loss: 0.9373733997344971, modularity_loss: -0.10213745385408401, purity_loss: 0.9633186459541321, regularization_loss: 0.07619219273328781
16:10:07 --- INFO: epoch: 461, batch: 1, loss: 0.9373131990432739, modularity_loss: -0.10212623327970505, purity_loss: 0.9632514715194702, regularization_loss: 0.07618793845176697
16:10:07 --- INFO: epoch: 462, batch: 1, loss: 0.937252402305603, modularity_loss: -0.10211454331874847, purity_loss: 0.9631835222244263, regularization_loss: 0.07618345320224762
16:10:08 --- INFO: epoch: 463, batch: 1, loss: 0.9371913075447083, modularity_loss: -0.10210242867469788, purity_loss: 0.9631149768829346, regularization_loss: 0.07617873698472977
16:10:08 --- INFO: epoch: 464, batch: 1, loss: 0.9371293783187866, modularity_loss: -0.10208996385335922, purity_loss: 0.963045597076416, regularization_loss: 0.07617375254631042
16:10:08 --- INFO: epoch: 465, batch: 1, loss: 0.9370667934417725, modularity_loss: -0.10207715630531311, purity_loss: 0.9629755616188049, regularization_loss: 0.07616841793060303
16:10:08 --- INFO: epoch: 466, batch: 1, loss: 0.9370032548904419, modularity_loss: -0.10206393152475357, purity_loss: 0.9629044532775879, regularization_loss: 0.07616270333528519
16:10:08 --- INFO: epoch: 467, batch: 1, loss: 0.9369388818740845, modularity_loss: -0.10205025225877762, purity_loss: 0.962832510471344, regularization_loss: 0.07615664601325989
16:10:08 --- INFO: epoch: 468, batch: 1, loss: 0.9368735551834106, modularity_loss: -0.10203605145215988, purity_loss: 0.9627593755722046, regularization_loss: 0.07615020126104355
16:10:08 --- INFO: epoch: 469, batch: 1, loss: 0.9368073344230652, modularity_loss: -0.10202132910490036, purity_loss: 0.9626852869987488, regularization_loss: 0.07614339888095856
16:10:08 --- INFO: epoch: 470, batch: 1, loss: 0.9367404580116272, modularity_loss: -0.10200599581003189, purity_loss: 0.9626101851463318, regularization_loss: 0.0761362612247467
16:10:08 --- INFO: epoch: 471, batch: 1, loss: 0.9366729855537415, modularity_loss: -0.10199007391929626, purity_loss: 0.9625341296195984, regularization_loss: 0.07612892985343933
16:10:08 --- INFO: epoch: 472, batch: 1, loss: 0.9366042613983154, modularity_loss: -0.1019735038280487, purity_loss: 0.9624564051628113, regularization_loss: 0.07612135261297226
16:10:08 --- INFO: epoch: 473, batch: 1, loss: 0.93653404712677, modularity_loss: -0.1019560694694519, purity_loss: 0.9623767137527466, regularization_loss: 0.07611338794231415
16:10:08 --- INFO: epoch: 474, batch: 1, loss: 0.9364629983901978, modularity_loss: -0.10193765163421631, purity_loss: 0.9622957110404968, regularization_loss: 0.07610496878623962
16:10:08 --- INFO: epoch: 475, batch: 1, loss: 0.9363903403282166, modularity_loss: -0.10191810876131058, purity_loss: 0.9622124433517456, regularization_loss: 0.07609599083662033
16:10:08 --- INFO: epoch: 476, batch: 1, loss: 0.9363161325454712, modularity_loss: -0.10189735144376755, purity_loss: 0.9621270298957825, regularization_loss: 0.07608644664287567
16:10:08 --- INFO: epoch: 477, batch: 1, loss: 0.9362398982048035, modularity_loss: -0.10187529027462006, purity_loss: 0.9620388746261597, regularization_loss: 0.07607633620500565
16:10:08 --- INFO: epoch: 478, batch: 1, loss: 0.9361622929573059, modularity_loss: -0.10185190290212631, purity_loss: 0.9619485139846802, regularization_loss: 0.07606569677591324
16:10:08 --- INFO: epoch: 479, batch: 1, loss: 0.936082124710083, modularity_loss: -0.10182710736989975, purity_loss: 0.9618546962738037, regularization_loss: 0.07605450600385666
16:10:08 --- INFO: epoch: 480, batch: 1, loss: 0.9359995126724243, modularity_loss: -0.1018008217215538, purity_loss: 0.9617576599121094, regularization_loss: 0.07604269683361053
16:10:08 --- INFO: epoch: 481, batch: 1, loss: 0.9359147548675537, modularity_loss: -0.10177294909954071, purity_loss: 0.9616575837135315, regularization_loss: 0.07603014260530472
16:10:08 --- INFO: epoch: 482, batch: 1, loss: 0.9358270764350891, modularity_loss: -0.10174334794282913, purity_loss: 0.961553692817688, regularization_loss: 0.07601674646139145
16:10:08 --- INFO: epoch: 483, batch: 1, loss: 0.9357365369796753, modularity_loss: -0.10171192139387131, purity_loss: 0.9614461064338684, regularization_loss: 0.07600237429141998
16:10:08 --- INFO: epoch: 484, batch: 1, loss: 0.9356423616409302, modularity_loss: -0.10167849808931351, purity_loss: 0.9613338708877563, regularization_loss: 0.07598696649074554
16:10:08 --- INFO: epoch: 485, batch: 1, loss: 0.9355445504188538, modularity_loss: -0.1016429141163826, purity_loss: 0.9612170457839966, regularization_loss: 0.07597042620182037
16:10:08 --- INFO: epoch: 486, batch: 1, loss: 0.9354422092437744, modularity_loss: -0.10160496085882187, purity_loss: 0.9610944986343384, regularization_loss: 0.0759526938199997
16:10:08 --- INFO: epoch: 487, batch: 1, loss: 0.9353354573249817, modularity_loss: -0.10156433284282684, purity_loss: 0.9609661102294922, regularization_loss: 0.07593368738889694
16:10:08 --- INFO: epoch: 488, batch: 1, loss: 0.9352232217788696, modularity_loss: -0.10152063518762589, purity_loss: 0.9608306288719177, regularization_loss: 0.07591322064399719
16:10:08 --- INFO: epoch: 489, batch: 1, loss: 0.9351049661636353, modularity_loss: -0.10147347301244736, purity_loss: 0.9606873393058777, regularization_loss: 0.07589109987020493
16:10:08 --- INFO: epoch: 490, batch: 1, loss: 0.9349800944328308, modularity_loss: -0.10142230242490768, purity_loss: 0.9605353474617004, regularization_loss: 0.07586704939603806
16:10:08 --- INFO: epoch: 491, batch: 1, loss: 0.9348472952842712, modularity_loss: -0.10136665403842926, purity_loss: 0.9603731036186218, regularization_loss: 0.07584082335233688
16:10:08 --- INFO: epoch: 492, batch: 1, loss: 0.9347055554389954, modularity_loss: -0.10130592435598373, purity_loss: 0.9601992964744568, regularization_loss: 0.07581216096878052
16:10:08 --- INFO: epoch: 493, batch: 1, loss: 0.9345536231994629, modularity_loss: -0.10123945772647858, purity_loss: 0.9600123167037964, regularization_loss: 0.07578075677156448
16:10:08 --- INFO: epoch: 494, batch: 1, loss: 0.9343901872634888, modularity_loss: -0.10116651654243469, purity_loss: 0.9598103761672974, regularization_loss: 0.07574629783630371
16:10:08 --- INFO: epoch: 495, batch: 1, loss: 0.9342130422592163, modularity_loss: -0.10108622908592224, purity_loss: 0.9595909118652344, regularization_loss: 0.07570835202932358
16:10:08 --- INFO: epoch: 496, batch: 1, loss: 0.9340198636054993, modularity_loss: -0.10099750012159348, purity_loss: 0.9593509435653687, regularization_loss: 0.0756664052605629
16:10:08 --- INFO: epoch: 497, batch: 1, loss: 0.9338082075119019, modularity_loss: -0.10089907795190811, purity_loss: 0.9590874910354614, regularization_loss: 0.07561979442834854
16:10:08 --- INFO: epoch: 498, batch: 1, loss: 0.9335743188858032, modularity_loss: -0.1007893905043602, purity_loss: 0.958795964717865, regularization_loss: 0.07556774467229843
16:10:08 --- INFO: epoch: 499, batch: 1, loss: 0.9333142638206482, modularity_loss: -0.10066650807857513, purity_loss: 0.958471417427063, regularization_loss: 0.07550934702157974
16:10:08 --- INFO: epoch: 500, batch: 1, loss: 0.9330229163169861, modularity_loss: -0.10052820295095444, purity_loss: 0.9581076502799988, regularization_loss: 0.07544344663619995
16:10:08 --- INFO: epoch: 501, batch: 1, loss: 0.9326933026313782, modularity_loss: -0.10037173330783844, purity_loss: 0.9576963782310486, regularization_loss: 0.07536865770816803
16:10:08 --- INFO: epoch: 502, batch: 1, loss: 0.9323176741600037, modularity_loss: -0.10019370168447495, purity_loss: 0.9572280645370483, regularization_loss: 0.07528328895568848
16:10:08 --- INFO: epoch: 503, batch: 1, loss: 0.9318854808807373, modularity_loss: -0.09998998790979385, purity_loss: 0.95669025182724, regularization_loss: 0.07518520951271057
16:10:08 --- INFO: epoch: 504, batch: 1, loss: 0.9313833713531494, modularity_loss: -0.09975535422563553, purity_loss: 0.956066906452179, regularization_loss: 0.07507182657718658
16:10:08 --- INFO: epoch: 505, batch: 1, loss: 0.9307942390441895, modularity_loss: -0.0994834378361702, purity_loss: 0.9553378224372864, regularization_loss: 0.07493983209133148
16:10:08 --- INFO: epoch: 506, batch: 1, loss: 0.9300958514213562, modularity_loss: -0.09916630387306213, purity_loss: 0.9544769525527954, regularization_loss: 0.07478521764278412
16:10:08 --- INFO: epoch: 507, batch: 1, loss: 0.9292596578598022, modularity_loss: -0.09879427403211594, purity_loss: 0.9534509778022766, regularization_loss: 0.07460296154022217
16:10:08 --- INFO: epoch: 508, batch: 1, loss: 0.9282487630844116, modularity_loss: -0.09835577756166458, purity_loss: 0.9522175192832947, regularization_loss: 0.07438699156045914
16:10:08 --- INFO: epoch: 509, batch: 1, loss: 0.9270164370536804, modularity_loss: -0.097837433218956, purity_loss: 0.9507237076759338, regularization_loss: 0.07413014024496078
16:10:08 --- INFO: epoch: 510, batch: 1, loss: 0.925504744052887, modularity_loss: -0.097224660217762, purity_loss: 0.9489049911499023, regularization_loss: 0.07382439076900482
16:10:08 --- INFO: epoch: 511, batch: 1, loss: 0.9236461520195007, modularity_loss: -0.09650380164384842, purity_loss: 0.9466881155967712, regularization_loss: 0.07346182316541672
16:10:08 --- INFO: epoch: 512, batch: 1, loss: 0.9213700890541077, modularity_loss: -0.09566642343997955, purity_loss: 0.9439998865127563, regularization_loss: 0.07303662598133087
16:10:08 --- INFO: epoch: 513, batch: 1, loss: 0.9186223745346069, modularity_loss: -0.09471739083528519, purity_loss: 0.940790593624115, regularization_loss: 0.07254917174577713
16:10:08 --- INFO: epoch: 514, batch: 1, loss: 0.9153952598571777, modularity_loss: -0.09368597716093063, purity_loss: 0.9370689988136292, regularization_loss: 0.0720122680068016
16:10:08 --- INFO: epoch: 515, batch: 1, loss: 0.9117718935012817, modularity_loss: -0.09263837337493896, purity_loss: 0.9329527020454407, regularization_loss: 0.07145756483078003
16:10:08 --- INFO: epoch: 516, batch: 1, loss: 0.9079582095146179, modularity_loss: -0.0916765108704567, purity_loss: 0.9286988973617554, regularization_loss: 0.07093584537506104
16:10:08 --- INFO: epoch: 517, batch: 1, loss: 0.9042091369628906, modularity_loss: -0.09091217815876007, purity_loss: 0.924615204334259, regularization_loss: 0.07050608098506927
16:10:08 --- INFO: epoch: 518, batch: 1, loss: 0.9006813168525696, modularity_loss: -0.09041538089513779, purity_loss: 0.9208900332450867, regularization_loss: 0.07020668685436249
16:10:08 --- INFO: epoch: 519, batch: 1, loss: 0.897223174571991, modularity_loss: -0.09017142653465271, purity_loss: 0.9173649549484253, regularization_loss: 0.07002962380647659
16:10:08 --- INFO: epoch: 520, batch: 1, loss: 0.8933529853820801, modularity_loss: -0.09009280800819397, purity_loss: 0.9135218858718872, regularization_loss: 0.06992389261722565
16:10:08 --- INFO: epoch: 521, batch: 1, loss: 0.8885039687156677, modularity_loss: -0.0900864526629448, purity_loss: 0.908759593963623, regularization_loss: 0.06983081996440887
16:10:08 --- INFO: epoch: 522, batch: 1, loss: 0.8823739290237427, modularity_loss: -0.09011386334896088, purity_loss: 0.9027674198150635, regularization_loss: 0.06972035765647888
16:10:08 --- INFO: epoch: 523, batch: 1, loss: 0.8750386238098145, modularity_loss: -0.09019879996776581, purity_loss: 0.895637035369873, regularization_loss: 0.069600410759449
16:10:08 --- INFO: epoch: 524, batch: 1, loss: 0.8670227527618408, modularity_loss: -0.09038027375936508, purity_loss: 0.8878993988037109, regularization_loss: 0.06950365751981735
16:10:08 --- INFO: epoch: 525, batch: 1, loss: 0.8590262532234192, modularity_loss: -0.09066522121429443, purity_loss: 0.8802335262298584, regularization_loss: 0.06945794820785522
16:10:08 --- INFO: epoch: 526, batch: 1, loss: 0.8515211343765259, modularity_loss: -0.0910184234380722, purity_loss: 0.8730683922767639, regularization_loss: 0.06947117298841476
16:10:08 --- INFO: epoch: 527, batch: 1, loss: 0.8446954488754272, modularity_loss: -0.09139128774404526, purity_loss: 0.866550862789154, regularization_loss: 0.06953586637973785
16:10:08 --- INFO: epoch: 528, batch: 1, loss: 0.838581383228302, modularity_loss: -0.09174571931362152, purity_loss: 0.8606858849525452, regularization_loss: 0.06964122503995895
16:10:08 --- INFO: epoch: 529, batch: 1, loss: 0.8331397175788879, modularity_loss: -0.09206467121839523, purity_loss: 0.855427622795105, regularization_loss: 0.06977677345275879
16:10:08 --- INFO: epoch: 530, batch: 1, loss: 0.8282365202903748, modularity_loss: -0.09233163297176361, purity_loss: 0.8506423234939575, regularization_loss: 0.06992580741643906
16:10:08 --- INFO: epoch: 531, batch: 1, loss: 0.8237207531929016, modularity_loss: -0.09253004193305969, purity_loss: 0.8461853861808777, regularization_loss: 0.07006540894508362
16:10:08 --- INFO: epoch: 532, batch: 1, loss: 0.8194476366043091, modularity_loss: -0.09264031052589417, purity_loss: 0.8419157862663269, regularization_loss: 0.07017213851213455
16:10:08 --- INFO: epoch: 533, batch: 1, loss: 0.8153335452079773, modularity_loss: -0.09265372902154922, purity_loss: 0.8377526998519897, regularization_loss: 0.07023459672927856
16:10:08 --- INFO: epoch: 534, batch: 1, loss: 0.8112906217575073, modularity_loss: -0.09258891642093658, purity_loss: 0.8336185216903687, regularization_loss: 0.07026102393865585
16:10:08 --- INFO: epoch: 535, batch: 1, loss: 0.8071581125259399, modularity_loss: -0.09249145537614822, purity_loss: 0.8293725252151489, regularization_loss: 0.07027706503868103
16:10:08 --- INFO: epoch: 536, batch: 1, loss: 0.8025994896888733, modularity_loss: -0.09240446239709854, purity_loss: 0.8246946334838867, regularization_loss: 0.07030930370092392
16:10:08 --- INFO: epoch: 537, batch: 1, loss: 0.7973613142967224, modularity_loss: -0.09235238283872604, purity_loss: 0.8193424940109253, regularization_loss: 0.07037120312452316
16:10:08 --- INFO: epoch: 538, batch: 1, loss: 0.7915810942649841, modularity_loss: -0.09233305603265762, purity_loss: 0.813456118106842, regularization_loss: 0.07045801728963852
16:10:08 --- INFO: epoch: 539, batch: 1, loss: 0.7858855724334717, modularity_loss: -0.09232252091169357, purity_loss: 0.8076551556587219, regularization_loss: 0.07055296748876572
16:10:08 --- INFO: epoch: 540, batch: 1, loss: 0.7812393307685852, modularity_loss: -0.09231611341238022, purity_loss: 0.8029108047485352, regularization_loss: 0.07064466178417206
16:10:08 --- INFO: epoch: 541, batch: 1, loss: 0.778679370880127, modularity_loss: -0.09232867509126663, purity_loss: 0.8002732992172241, regularization_loss: 0.07073475420475006
16:10:08 --- INFO: epoch: 542, batch: 1, loss: 0.7784007787704468, modularity_loss: -0.0923662781715393, purity_loss: 0.7999424338340759, regularization_loss: 0.07082464545965195
16:10:08 --- INFO: epoch: 543, batch: 1, loss: 0.7792588472366333, modularity_loss: -0.0924207791686058, purity_loss: 0.8007784485816956, regularization_loss: 0.07090115547180176
16:10:08 --- INFO: epoch: 544, batch: 1, loss: 0.7795255184173584, modularity_loss: -0.09247878938913345, purity_loss: 0.801058828830719, regularization_loss: 0.07094546407461166
16:10:08 --- INFO: epoch: 545, batch: 1, loss: 0.7781104445457458, modularity_loss: -0.09253677725791931, purity_loss: 0.799697756767273, regularization_loss: 0.0709494799375534
16:10:08 --- INFO: epoch: 546, batch: 1, loss: 0.7748968601226807, modularity_loss: -0.09258930385112762, purity_loss: 0.7965707182884216, regularization_loss: 0.07091544568538666
16:10:08 --- INFO: epoch: 547, batch: 1, loss: 0.7705934047698975, modularity_loss: -0.0926271378993988, purity_loss: 0.7923685908317566, regularization_loss: 0.07085194438695908
16:10:08 --- INFO: epoch: 548, batch: 1, loss: 0.766326904296875, modularity_loss: -0.09265750646591187, purity_loss: 0.7882069945335388, regularization_loss: 0.07077743858098984
16:10:08 --- INFO: epoch: 549, batch: 1, loss: 0.7631054520606995, modularity_loss: -0.09270192682743073, purity_loss: 0.7850910425186157, regularization_loss: 0.07071632146835327
16:10:08 --- INFO: epoch: 550, batch: 1, loss: 0.7611908316612244, modularity_loss: -0.09277411550283432, purity_loss: 0.7832813262939453, regularization_loss: 0.07068359851837158
16:10:08 --- INFO: epoch: 551, batch: 1, loss: 0.7602494359016418, modularity_loss: -0.09286278486251831, purity_loss: 0.7824352383613586, regularization_loss: 0.07067696750164032
16:10:08 --- INFO: epoch: 552, batch: 1, loss: 0.759861409664154, modularity_loss: -0.09293816238641739, purity_loss: 0.7821150422096252, regularization_loss: 0.0706845074892044
16:10:08 --- INFO: epoch: 553, batch: 1, loss: 0.7595971822738647, modularity_loss: -0.09298326820135117, purity_loss: 0.7818880081176758, regularization_loss: 0.07069246470928192
16:10:08 --- INFO: epoch: 554, batch: 1, loss: 0.7589698433876038, modularity_loss: -0.09299437701702118, purity_loss: 0.7812734842300415, regularization_loss: 0.07069074362516403
16:10:08 --- INFO: epoch: 555, batch: 1, loss: 0.7578413486480713, modularity_loss: -0.09298349916934967, purity_loss: 0.7801455855369568, regularization_loss: 0.07067926973104477
16:10:08 --- INFO: epoch: 556, batch: 1, loss: 0.7564858794212341, modularity_loss: -0.09297291934490204, purity_loss: 0.7787913084030151, regularization_loss: 0.07066746801137924
16:10:08 --- INFO: epoch: 557, batch: 1, loss: 0.7551385164260864, modularity_loss: -0.09298494458198547, purity_loss: 0.7774562239646912, regularization_loss: 0.07066725194454193
16:10:08 --- INFO: epoch: 558, batch: 1, loss: 0.7537966966629028, modularity_loss: -0.09302884340286255, purity_loss: 0.7761411666870117, regularization_loss: 0.07068437337875366
16:10:08 --- INFO: epoch: 559, batch: 1, loss: 0.7525890469551086, modularity_loss: -0.09310154616832733, purity_loss: 0.7749744653701782, regularization_loss: 0.07071612775325775
16:10:08 --- INFO: epoch: 560, batch: 1, loss: 0.751763105392456, modularity_loss: -0.0931854248046875, purity_loss: 0.7741941213607788, regularization_loss: 0.07075443863868713
16:10:08 --- INFO: epoch: 561, batch: 1, loss: 0.7513659000396729, modularity_loss: -0.09325931966304779, purity_loss: 0.773836076259613, regularization_loss: 0.07078913599252701
16:10:08 --- INFO: epoch: 562, batch: 1, loss: 0.7512048482894897, modularity_loss: -0.09330965578556061, purity_loss: 0.7737016081809998, regularization_loss: 0.07081292569637299
16:10:08 --- INFO: epoch: 563, batch: 1, loss: 0.7510465383529663, modularity_loss: -0.09333223104476929, purity_loss: 0.773554801940918, regularization_loss: 0.07082397490739822
16:10:08 --- INFO: epoch: 564, batch: 1, loss: 0.7508033514022827, modularity_loss: -0.09333311766386032, purity_loss: 0.7733117341995239, regularization_loss: 0.07082471251487732
16:10:08 --- INFO: epoch: 565, batch: 1, loss: 0.7504197359085083, modularity_loss: -0.09332088381052017, purity_loss: 0.7729209065437317, regularization_loss: 0.07081973552703857
16:10:08 --- INFO: epoch: 566, batch: 1, loss: 0.7499008178710938, modularity_loss: -0.09330801665782928, purity_loss: 0.7723954916000366, regularization_loss: 0.07081333547830582
16:10:08 --- INFO: epoch: 567, batch: 1, loss: 0.7493342757225037, modularity_loss: -0.09330308437347412, purity_loss: 0.7718294858932495, regularization_loss: 0.07080785185098648
16:10:08 --- INFO: epoch: 568, batch: 1, loss: 0.7488137483596802, modularity_loss: -0.0933084487915039, purity_loss: 0.7713198661804199, regularization_loss: 0.07080234587192535
16:10:09 --- INFO: epoch: 569, batch: 1, loss: 0.7483699917793274, modularity_loss: -0.09332047402858734, purity_loss: 0.7708960175514221, regularization_loss: 0.07079444080591202
16:10:09 --- INFO: epoch: 570, batch: 1, loss: 0.7480035424232483, modularity_loss: -0.09333717823028564, purity_loss: 0.7705568671226501, regularization_loss: 0.07078386843204498
16:10:09 --- INFO: epoch: 571, batch: 1, loss: 0.7477564215660095, modularity_loss: -0.09335807710886002, purity_loss: 0.7703410983085632, regularization_loss: 0.07077338546514511
16:10:09 --- INFO: epoch: 572, batch: 1, loss: 0.747608482837677, modularity_loss: -0.09338071942329407, purity_loss: 0.7702234983444214, regularization_loss: 0.07076568156480789
16:10:09 --- INFO: epoch: 573, batch: 1, loss: 0.7474679946899414, modularity_loss: -0.09340198338031769, purity_loss: 0.7701082229614258, regularization_loss: 0.07076174020767212
16:10:09 --- INFO: epoch: 574, batch: 1, loss: 0.7472779750823975, modularity_loss: -0.09341887384653091, purity_loss: 0.7699357867240906, regularization_loss: 0.0707610473036766
16:10:09 --- INFO: epoch: 575, batch: 1, loss: 0.7470466494560242, modularity_loss: -0.09343061596155167, purity_loss: 0.7697148323059082, regularization_loss: 0.07076241821050644
16:10:09 --- INFO: epoch: 576, batch: 1, loss: 0.7467981576919556, modularity_loss: -0.09343813359737396, purity_loss: 0.7694722414016724, regularization_loss: 0.07076407968997955
16:10:09 --- INFO: epoch: 577, batch: 1, loss: 0.7465366125106812, modularity_loss: -0.09344257414340973, purity_loss: 0.7692146301269531, regularization_loss: 0.07076455652713776
16:10:09 --- INFO: epoch: 578, batch: 1, loss: 0.7462625503540039, modularity_loss: -0.09344581514596939, purity_loss: 0.768944501876831, regularization_loss: 0.07076387107372284
16:10:09 --- INFO: epoch: 579, batch: 1, loss: 0.7459918260574341, modularity_loss: -0.09345054626464844, purity_loss: 0.7686783075332642, regularization_loss: 0.07076405733823776
16:10:09 --- INFO: epoch: 580, batch: 1, loss: 0.7457479238510132, modularity_loss: -0.09345874935388565, purity_loss: 0.7684391140937805, regularization_loss: 0.07076752930879593
16:10:09 --- INFO: epoch: 581, batch: 1, loss: 0.7455375790596008, modularity_loss: -0.0934704914689064, purity_loss: 0.7682327032089233, regularization_loss: 0.07077536731958389
16:10:09 --- INFO: epoch: 582, batch: 1, loss: 0.7453617453575134, modularity_loss: -0.09348340332508087, purity_loss: 0.7680585980415344, regularization_loss: 0.07078657299280167
16:10:09 --- INFO: epoch: 583, batch: 1, loss: 0.7452112436294556, modularity_loss: -0.0934939831495285, purity_loss: 0.7679066061973572, regularization_loss: 0.0707985907793045
16:10:09 --- INFO: epoch: 584, batch: 1, loss: 0.7450659275054932, modularity_loss: -0.09349940717220306, purity_loss: 0.7677568793296814, regularization_loss: 0.07080844789743423
16:10:09 --- INFO: epoch: 585, batch: 1, loss: 0.7449102401733398, modularity_loss: -0.09349925071001053, purity_loss: 0.7675955295562744, regularization_loss: 0.07081398367881775
16:10:09 --- INFO: epoch: 586, batch: 1, loss: 0.7447464466094971, modularity_loss: -0.09349571168422699, purity_loss: 0.7674276828765869, regularization_loss: 0.07081449776887894
16:10:09 --- INFO: epoch: 587, batch: 1, loss: 0.7445830702781677, modularity_loss: -0.09349187463521957, purity_loss: 0.7672640085220337, regularization_loss: 0.07081093639135361
16:10:09 --- INFO: epoch: 588, batch: 1, loss: 0.7444175481796265, modularity_loss: -0.09349101781845093, purity_loss: 0.7671033143997192, regularization_loss: 0.07080524414777756
16:10:09 --- INFO: epoch: 589, batch: 1, loss: 0.7442484498023987, modularity_loss: -0.09349513798952103, purity_loss: 0.766944169998169, regularization_loss: 0.07079940289258957
16:10:09 --- INFO: epoch: 590, batch: 1, loss: 0.7440831065177917, modularity_loss: -0.0935041680932045, purity_loss: 0.7667924761772156, regularization_loss: 0.07079482078552246
16:10:09 --- INFO: epoch: 591, batch: 1, loss: 0.743933379650116, modularity_loss: -0.09351573884487152, purity_loss: 0.7666573524475098, regularization_loss: 0.07079177349805832
16:10:09 --- INFO: epoch: 592, batch: 1, loss: 0.743798017501831, modularity_loss: -0.0935269296169281, purity_loss: 0.7665348649024963, regularization_loss: 0.07079008221626282
16:10:09 --- INFO: epoch: 593, batch: 1, loss: 0.7436715364456177, modularity_loss: -0.09353531152009964, purity_loss: 0.7664176225662231, regularization_loss: 0.07078924030065536
16:10:09 --- INFO: epoch: 594, batch: 1, loss: 0.7435483932495117, modularity_loss: -0.09353950619697571, purity_loss: 0.7662989497184753, regularization_loss: 0.07078897207975388
16:10:09 --- INFO: epoch: 595, batch: 1, loss: 0.7434263229370117, modularity_loss: -0.09353981912136078, purity_loss: 0.7661769986152649, regularization_loss: 0.0707891434431076
16:10:09 --- INFO: epoch: 596, batch: 1, loss: 0.7433040142059326, modularity_loss: -0.09353826195001602, purity_loss: 0.7660527229309082, regularization_loss: 0.07078958302736282
16:10:09 --- INFO: epoch: 597, batch: 1, loss: 0.7431795597076416, modularity_loss: -0.09353629499673843, purity_loss: 0.7659255862236023, regularization_loss: 0.07079028338193893
16:10:09 --- INFO: epoch: 598, batch: 1, loss: 0.7430543303489685, modularity_loss: -0.09353499859571457, purity_loss: 0.7657982707023621, regularization_loss: 0.0707910805940628
16:10:09 --- INFO: epoch: 599, batch: 1, loss: 0.742929220199585, modularity_loss: -0.09353494644165039, purity_loss: 0.7656723260879517, regularization_loss: 0.07079184800386429
16:10:09 --- INFO: epoch: 600, batch: 1, loss: 0.7428048849105835, modularity_loss: -0.09353616833686829, purity_loss: 0.7655484676361084, regularization_loss: 0.07079258561134338
16:10:09 --- INFO: epoch: 601, batch: 1, loss: 0.7426831126213074, modularity_loss: -0.09353843331336975, purity_loss: 0.7654279470443726, regularization_loss: 0.07079358398914337
16:10:09 --- INFO: epoch: 602, batch: 1, loss: 0.7425664663314819, modularity_loss: -0.09354142844676971, purity_loss: 0.765312671661377, regularization_loss: 0.07079519331455231
16:10:09 --- INFO: epoch: 603, batch: 1, loss: 0.742455244064331, modularity_loss: -0.09354451298713684, purity_loss: 0.7652022242546082, regularization_loss: 0.07079754024744034
16:10:09 --- INFO: epoch: 604, batch: 1, loss: 0.7423467040061951, modularity_loss: -0.09354694187641144, purity_loss: 0.7650933265686035, regularization_loss: 0.0708003044128418
16:10:09 --- INFO: epoch: 605, batch: 1, loss: 0.7422394156455994, modularity_loss: -0.09354807436466217, purity_loss: 0.7649846076965332, regularization_loss: 0.07080286741256714
16:10:09 --- INFO: epoch: 606, batch: 1, loss: 0.7421324849128723, modularity_loss: -0.09354756027460098, purity_loss: 0.7648754715919495, regularization_loss: 0.07080456614494324
16:10:09 --- INFO: epoch: 607, batch: 1, loss: 0.7420259714126587, modularity_loss: -0.09354554861783981, purity_loss: 0.7647665739059448, regularization_loss: 0.07080495357513428
16:10:09 --- INFO: epoch: 608, batch: 1, loss: 0.7419204711914062, modularity_loss: -0.0935426577925682, purity_loss: 0.7646591067314148, regularization_loss: 0.07080404460430145
16:10:09 --- INFO: epoch: 609, batch: 1, loss: 0.7418143153190613, modularity_loss: -0.09353893250226974, purity_loss: 0.7645506262779236, regularization_loss: 0.07080260664224625
16:10:09 --- INFO: epoch: 610, batch: 1, loss: 0.7417088747024536, modularity_loss: -0.09353569149971008, purity_loss: 0.7644432187080383, regularization_loss: 0.07080136984586716
16:10:09 --- INFO: epoch: 611, batch: 1, loss: 0.7416046261787415, modularity_loss: -0.09353403002023697, purity_loss: 0.7643377184867859, regularization_loss: 0.07080096006393433
16:10:09 --- INFO: epoch: 612, batch: 1, loss: 0.7415021657943726, modularity_loss: -0.09353428333997726, purity_loss: 0.7642349600791931, regularization_loss: 0.0708014965057373
16:10:09 --- INFO: epoch: 613, batch: 1, loss: 0.7414019703865051, modularity_loss: -0.09353569149971008, purity_loss: 0.7641352415084839, regularization_loss: 0.07080242782831192
16:10:09 --- INFO: epoch: 614, batch: 1, loss: 0.7413040995597839, modularity_loss: -0.09353736788034439, purity_loss: 0.7640383243560791, regularization_loss: 0.07080312818288803
16:10:09 --- INFO: epoch: 615, batch: 1, loss: 0.7412075996398926, modularity_loss: -0.09353845566511154, purity_loss: 0.7639430165290833, regularization_loss: 0.07080304622650146
16:10:09 --- INFO: epoch: 616, batch: 1, loss: 0.7411117553710938, modularity_loss: -0.09353867918252945, purity_loss: 0.7638483643531799, regularization_loss: 0.07080206274986267
16:10:09 --- INFO: epoch: 617, batch: 1, loss: 0.7410163283348083, modularity_loss: -0.09353821724653244, purity_loss: 0.7637540698051453, regularization_loss: 0.07080046832561493
16:10:09 --- INFO: epoch: 618, batch: 1, loss: 0.7409213781356812, modularity_loss: -0.0935375839471817, purity_loss: 0.7636601328849792, regularization_loss: 0.07079882174730301
16:10:09 --- INFO: epoch: 619, batch: 1, loss: 0.7408269047737122, modularity_loss: -0.09353729337453842, purity_loss: 0.7635664343833923, regularization_loss: 0.07079775631427765
16:10:09 --- INFO: epoch: 620, batch: 1, loss: 0.7407326102256775, modularity_loss: -0.09353754669427872, purity_loss: 0.7634725570678711, regularization_loss: 0.07079761475324631
16:10:09 --- INFO: epoch: 621, batch: 1, loss: 0.7406389117240906, modularity_loss: -0.09353812783956528, purity_loss: 0.7633786797523499, regularization_loss: 0.0707983672618866
16:10:09 --- INFO: epoch: 622, batch: 1, loss: 0.7405459880828857, modularity_loss: -0.09353858977556229, purity_loss: 0.7632848024368286, regularization_loss: 0.0707997977733612
16:10:09 --- INFO: epoch: 623, batch: 1, loss: 0.7404538989067078, modularity_loss: -0.09353847801685333, purity_loss: 0.7631909251213074, regularization_loss: 0.07080145180225372
16:10:09 --- INFO: epoch: 624, batch: 1, loss: 0.7403627634048462, modularity_loss: -0.09353724122047424, purity_loss: 0.7630968689918518, regularization_loss: 0.07080314308404922
16:10:09 --- INFO: epoch: 625, batch: 1, loss: 0.7402726411819458, modularity_loss: -0.09353506565093994, purity_loss: 0.7630030512809753, regularization_loss: 0.0708046481013298
16:10:09 --- INFO: epoch: 626, batch: 1, loss: 0.7401832342147827, modularity_loss: -0.09353242069482803, purity_loss: 0.7629097700119019, regularization_loss: 0.07080590724945068
16:10:09 --- INFO: epoch: 627, batch: 1, loss: 0.7400941848754883, modularity_loss: -0.09352986514568329, purity_loss: 0.7628171443939209, regularization_loss: 0.07080692052841187
16:10:09 --- INFO: epoch: 628, batch: 1, loss: 0.7400059700012207, modularity_loss: -0.09352786093950272, purity_loss: 0.7627262473106384, regularization_loss: 0.07080761343240738
16:10:09 --- INFO: epoch: 629, batch: 1, loss: 0.7399182319641113, modularity_loss: -0.09352651983499527, purity_loss: 0.7626368403434753, regularization_loss: 0.07080789655447006
16:10:09 --- INFO: epoch: 630, batch: 1, loss: 0.7398306131362915, modularity_loss: -0.0935259684920311, purity_loss: 0.7625488638877869, regularization_loss: 0.07080774754285812
16:10:09 --- INFO: epoch: 631, batch: 1, loss: 0.7397432923316956, modularity_loss: -0.09352611005306244, purity_loss: 0.7624620795249939, regularization_loss: 0.07080730050802231
16:10:09 --- INFO: epoch: 632, batch: 1, loss: 0.7396563291549683, modularity_loss: -0.09352676570415497, purity_loss: 0.7623763680458069, regularization_loss: 0.07080670446157455
16:10:09 --- INFO: epoch: 633, batch: 1, loss: 0.7395696640014648, modularity_loss: -0.0935276448726654, purity_loss: 0.7622911930084229, regularization_loss: 0.07080614566802979
16:10:09 --- INFO: epoch: 634, batch: 1, loss: 0.7394835948944092, modularity_loss: -0.0935283824801445, purity_loss: 0.7622061967849731, regularization_loss: 0.07080578804016113
16:10:09 --- INFO: epoch: 635, batch: 1, loss: 0.7393982410430908, modularity_loss: -0.0935286432504654, purity_loss: 0.7621212601661682, regularization_loss: 0.0708056092262268
16:10:09 --- INFO: epoch: 636, batch: 1, loss: 0.7393131256103516, modularity_loss: -0.09352812170982361, purity_loss: 0.762035608291626, regularization_loss: 0.0708056092262268
16:10:09 --- INFO: epoch: 637, batch: 1, loss: 0.7392284870147705, modularity_loss: -0.09352683275938034, purity_loss: 0.7619495987892151, regularization_loss: 0.07080571353435516
16:10:09 --- INFO: epoch: 638, batch: 1, loss: 0.7391442060470581, modularity_loss: -0.09352496266365051, purity_loss: 0.7618632316589355, regularization_loss: 0.07080590724945068
16:10:09 --- INFO: epoch: 639, batch: 1, loss: 0.73906010389328, modularity_loss: -0.09352283924818039, purity_loss: 0.7617767453193665, regularization_loss: 0.07080620527267456
16:10:09 --- INFO: epoch: 640, batch: 1, loss: 0.7389763593673706, modularity_loss: -0.0935208648443222, purity_loss: 0.761690616607666, regularization_loss: 0.07080663740634918
16:10:09 --- INFO: epoch: 641, batch: 1, loss: 0.7388930320739746, modularity_loss: -0.09351932257413864, purity_loss: 0.7616051435470581, regularization_loss: 0.07080721855163574
16:10:09 --- INFO: epoch: 642, batch: 1, loss: 0.7388099431991577, modularity_loss: -0.09351836144924164, purity_loss: 0.7615204453468323, regularization_loss: 0.07080788165330887
16:10:09 --- INFO: epoch: 643, batch: 1, loss: 0.738727331161499, modularity_loss: -0.09351766109466553, purity_loss: 0.7614365220069885, regularization_loss: 0.07080846279859543
16:10:09 --- INFO: epoch: 644, batch: 1, loss: 0.7386450171470642, modularity_loss: -0.09351710975170135, purity_loss: 0.7613533139228821, regularization_loss: 0.07080882787704468
16:10:09 --- INFO: epoch: 645, batch: 1, loss: 0.738563060760498, modularity_loss: -0.09351644665002823, purity_loss: 0.7612705230712891, regularization_loss: 0.07080897688865662
16:10:09 --- INFO: epoch: 646, batch: 1, loss: 0.7384815812110901, modularity_loss: -0.093515545129776, purity_loss: 0.7611882090568542, regularization_loss: 0.07080892473459244
16:10:09 --- INFO: epoch: 647, batch: 1, loss: 0.7384002804756165, modularity_loss: -0.09351447224617004, purity_loss: 0.7611059546470642, regularization_loss: 0.07080880552530289
16:10:09 --- INFO: epoch: 648, batch: 1, loss: 0.7383193969726562, modularity_loss: -0.09351326525211334, purity_loss: 0.7610239386558533, regularization_loss: 0.07080873847007751
16:10:09 --- INFO: epoch: 649, batch: 1, loss: 0.7382385730743408, modularity_loss: -0.09351208060979843, purity_loss: 0.7609418034553528, regularization_loss: 0.07080882042646408
16:10:09 --- INFO: epoch: 650, batch: 1, loss: 0.738158106803894, modularity_loss: -0.0935109555721283, purity_loss: 0.7608599662780762, regularization_loss: 0.07080907374620438
16:10:09 --- INFO: epoch: 651, batch: 1, loss: 0.7380779981613159, modularity_loss: -0.09350987523794174, purity_loss: 0.7607784271240234, regularization_loss: 0.07080941647291183
16:10:09 --- INFO: epoch: 652, batch: 1, loss: 0.7379980087280273, modularity_loss: -0.09350880235433578, purity_loss: 0.760697066783905, regularization_loss: 0.0708097442984581
16:10:09 --- INFO: epoch: 653, batch: 1, loss: 0.7379181981086731, modularity_loss: -0.09350767731666565, purity_loss: 0.760615885257721, regularization_loss: 0.07080996781587601
16:10:09 --- INFO: epoch: 654, batch: 1, loss: 0.7378387451171875, modularity_loss: -0.09350647777318954, purity_loss: 0.7605351805686951, regularization_loss: 0.07081004977226257
16:10:09 --- INFO: epoch: 655, batch: 1, loss: 0.737759530544281, modularity_loss: -0.09350524097681046, purity_loss: 0.7604547739028931, regularization_loss: 0.07081001996994019
16:10:09 --- INFO: epoch: 656, batch: 1, loss: 0.7376804947853088, modularity_loss: -0.09350403398275375, purity_loss: 0.7603745460510254, regularization_loss: 0.07080996781587601
16:10:09 --- INFO: epoch: 657, batch: 1, loss: 0.7376017570495605, modularity_loss: -0.09350286424160004, purity_loss: 0.7602947354316711, regularization_loss: 0.07080990821123123
16:10:09 --- INFO: epoch: 658, batch: 1, loss: 0.7375231385231018, modularity_loss: -0.09350176155567169, purity_loss: 0.7602149844169617, regularization_loss: 0.07080990821123123
16:10:09 --- INFO: epoch: 659, batch: 1, loss: 0.7374446988105774, modularity_loss: -0.09350073337554932, purity_loss: 0.7601354718208313, regularization_loss: 0.07080995291471481
16:10:09 --- INFO: epoch: 660, batch: 1, loss: 0.7373665571212769, modularity_loss: -0.09349977970123291, purity_loss: 0.7600563168525696, regularization_loss: 0.07081003487110138
16:10:09 --- INFO: epoch: 661, batch: 1, loss: 0.7372885942459106, modularity_loss: -0.09349879622459412, purity_loss: 0.7599772214889526, regularization_loss: 0.07081016898155212
16:10:09 --- INFO: epoch: 662, batch: 1, loss: 0.7372106313705444, modularity_loss: -0.09349779784679413, purity_loss: 0.7598981261253357, regularization_loss: 0.07081031799316406
16:10:09 --- INFO: epoch: 663, batch: 1, loss: 0.7371329069137573, modularity_loss: -0.09349669516086578, purity_loss: 0.7598191499710083, regularization_loss: 0.0708104595541954
16:10:09 --- INFO: epoch: 664, batch: 1, loss: 0.7370555400848389, modularity_loss: -0.09349550306797028, purity_loss: 0.75974041223526, regularization_loss: 0.07081061601638794
16:10:09 --- INFO: epoch: 665, batch: 1, loss: 0.7369781732559204, modularity_loss: -0.0934942290186882, purity_loss: 0.7596616148948669, regularization_loss: 0.07081077247858047
16:10:09 --- INFO: epoch: 666, batch: 1, loss: 0.7369009852409363, modularity_loss: -0.09349286556243896, purity_loss: 0.7595829367637634, regularization_loss: 0.07081093639135361
16:10:09 --- INFO: epoch: 667, batch: 1, loss: 0.7368240356445312, modularity_loss: -0.09349150955677032, purity_loss: 0.7595044374465942, regularization_loss: 0.07081109285354614
16:10:09 --- INFO: epoch: 668, batch: 1, loss: 0.7367472052574158, modularity_loss: -0.09349016845226288, purity_loss: 0.7594261169433594, regularization_loss: 0.07081124931573868
16:10:09 --- INFO: epoch: 669, batch: 1, loss: 0.7366705536842346, modularity_loss: -0.09348893165588379, purity_loss: 0.7593480944633484, regularization_loss: 0.07081139832735062
16:10:09 --- INFO: epoch: 670, batch: 1, loss: 0.7365939617156982, modularity_loss: -0.09348783642053604, purity_loss: 0.7592703104019165, regularization_loss: 0.07081150263547897
16:10:09 --- INFO: epoch: 671, batch: 1, loss: 0.736517608165741, modularity_loss: -0.09348684549331665, purity_loss: 0.7591928839683533, regularization_loss: 0.07081158459186554
16:10:09 --- INFO: epoch: 672, batch: 1, loss: 0.7364413142204285, modularity_loss: -0.09348592162132263, purity_loss: 0.7591156363487244, regularization_loss: 0.07081161439418793
16:10:09 --- INFO: epoch: 673, batch: 1, loss: 0.7363651394844055, modularity_loss: -0.0934850350022316, purity_loss: 0.7590385675430298, regularization_loss: 0.07081159949302673
16:10:09 --- INFO: epoch: 674, batch: 1, loss: 0.7362890243530273, modularity_loss: -0.09348412603139877, purity_loss: 0.7589616179466248, regularization_loss: 0.07081156224012375
16:10:10 --- INFO: epoch: 675, batch: 1, loss: 0.7362130284309387, modularity_loss: -0.09348315000534058, purity_loss: 0.758884608745575, regularization_loss: 0.07081154733896255
16:10:10 --- INFO: epoch: 676, batch: 1, loss: 0.7361369132995605, modularity_loss: -0.0934821218252182, purity_loss: 0.7588074803352356, regularization_loss: 0.07081157714128494
16:10:10 --- INFO: epoch: 677, batch: 1, loss: 0.7360609769821167, modularity_loss: -0.09348100423812866, purity_loss: 0.7587303519248962, regularization_loss: 0.07081163674592972
16:10:10 --- INFO: epoch: 678, batch: 1, loss: 0.7359851002693176, modularity_loss: -0.09347983449697495, purity_loss: 0.7586532235145569, regularization_loss: 0.07081171870231628
16:10:10 --- INFO: epoch: 679, batch: 1, loss: 0.7359094023704529, modularity_loss: -0.09347864240407944, purity_loss: 0.7585762143135071, regularization_loss: 0.07081180810928345
16:10:10 --- INFO: epoch: 680, batch: 1, loss: 0.7358337640762329, modularity_loss: -0.09347746521234512, purity_loss: 0.7584993243217468, regularization_loss: 0.07081188261508942
16:10:10 --- INFO: epoch: 681, batch: 1, loss: 0.7357581257820129, modularity_loss: -0.0934763103723526, purity_loss: 0.7584224939346313, regularization_loss: 0.07081194221973419
16:10:10 --- INFO: epoch: 682, batch: 1, loss: 0.7356828451156616, modularity_loss: -0.09347523003816605, purity_loss: 0.7583460807800293, regularization_loss: 0.07081197202205658
16:10:10 --- INFO: epoch: 683, batch: 1, loss: 0.7356073260307312, modularity_loss: -0.09347419440746307, purity_loss: 0.7582695484161377, regularization_loss: 0.07081197202205658
16:10:10 --- INFO: epoch: 684, batch: 1, loss: 0.7355320453643799, modularity_loss: -0.09347321093082428, purity_loss: 0.75819331407547, regularization_loss: 0.07081197202205658
16:10:10 --- INFO: epoch: 685, batch: 1, loss: 0.7354568243026733, modularity_loss: -0.09347228705883026, purity_loss: 0.7581170797348022, regularization_loss: 0.07081200927495956
16:10:10 --- INFO: epoch: 686, batch: 1, loss: 0.7353820204734802, modularity_loss: -0.09347137808799744, purity_loss: 0.7580413818359375, regularization_loss: 0.07081203907728195
16:10:10 --- INFO: epoch: 687, batch: 1, loss: 0.7353072166442871, modularity_loss: -0.0934712141752243, purity_loss: 0.7579665780067444, regularization_loss: 0.07081184536218643
16:10:10 --- INFO: epoch: 688, batch: 1, loss: 0.735232412815094, modularity_loss: -0.09347148984670639, purity_loss: 0.7578924298286438, regularization_loss: 0.07081146538257599
16:10:10 --- INFO: epoch: 689, batch: 1, loss: 0.7351576685905457, modularity_loss: -0.09347176551818848, purity_loss: 0.7578184604644775, regularization_loss: 0.07081099599599838
16:10:10 --- INFO: epoch: 690, batch: 1, loss: 0.7350832223892212, modularity_loss: -0.09347167611122131, purity_loss: 0.757744312286377, regularization_loss: 0.07081057131290436
16:10:10 --- INFO: epoch: 691, batch: 1, loss: 0.735008716583252, modularity_loss: -0.0934709832072258, purity_loss: 0.7576694488525391, regularization_loss: 0.07081025838851929
16:10:10 --- INFO: epoch: 692, batch: 1, loss: 0.7349339723587036, modularity_loss: -0.09346889704465866, purity_loss: 0.7575924396514893, regularization_loss: 0.07081044465303421
16:10:10 --- INFO: epoch: 693, batch: 1, loss: 0.7348594069480896, modularity_loss: -0.0934666246175766, purity_loss: 0.7575152516365051, regularization_loss: 0.07081079483032227
16:10:10 --- INFO: epoch: 694, batch: 1, loss: 0.7347850799560547, modularity_loss: -0.09346451610326767, purity_loss: 0.7574383616447449, regularization_loss: 0.0708112046122551
16:10:10 --- INFO: epoch: 695, batch: 1, loss: 0.734710693359375, modularity_loss: -0.09346279501914978, purity_loss: 0.7573619484901428, regularization_loss: 0.07081156969070435
16:10:10 --- INFO: epoch: 696, batch: 1, loss: 0.7346364855766296, modularity_loss: -0.09346148371696472, purity_loss: 0.7572862505912781, regularization_loss: 0.07081172615289688
16:10:10 --- INFO: epoch: 697, batch: 1, loss: 0.734562337398529, modularity_loss: -0.09345973283052444, purity_loss: 0.7572101950645447, regularization_loss: 0.07081187516450882
16:10:10 --- INFO: epoch: 698, batch: 1, loss: 0.7344881296157837, modularity_loss: -0.09345777332782745, purity_loss: 0.7571339011192322, regularization_loss: 0.07081198692321777
16:10:10 --- INFO: epoch: 699, batch: 1, loss: 0.7344141006469727, modularity_loss: -0.09345588088035583, purity_loss: 0.7570579051971436, regularization_loss: 0.07081207633018494
16:10:10 --- INFO: epoch: 700, batch: 1, loss: 0.7343399524688721, modularity_loss: -0.09345515817403793, purity_loss: 0.75698322057724, regularization_loss: 0.07081187516450882
16:10:10 --- INFO: epoch: 701, batch: 1, loss: 0.7342658042907715, modularity_loss: -0.09345459938049316, purity_loss: 0.7569087147712708, regularization_loss: 0.0708116963505745
16:10:10 --- INFO: epoch: 702, batch: 1, loss: 0.7341917157173157, modularity_loss: -0.09345412999391556, purity_loss: 0.7568343281745911, regularization_loss: 0.07081151008605957
16:10:10 --- INFO: epoch: 703, batch: 1, loss: 0.7341176271438599, modularity_loss: -0.09345439076423645, purity_loss: 0.7567609548568726, regularization_loss: 0.07081104069948196
16:10:10 --- INFO: epoch: 704, batch: 1, loss: 0.7340435981750488, modularity_loss: -0.09345418959856033, purity_loss: 0.7566871643066406, regularization_loss: 0.07081065326929092
16:10:10 --- INFO: epoch: 705, batch: 1, loss: 0.7339695692062378, modularity_loss: -0.09345345199108124, purity_loss: 0.7566125988960266, regularization_loss: 0.07081042230129242
16:10:10 --- INFO: epoch: 706, batch: 1, loss: 0.7338955998420715, modularity_loss: -0.09345301985740662, purity_loss: 0.756538450717926, regularization_loss: 0.07081016898155212
16:10:10 --- INFO: epoch: 707, batch: 1, loss: 0.7338218688964844, modularity_loss: -0.0934528335928917, purity_loss: 0.7564647197723389, regularization_loss: 0.07080996781587601
16:10:10 --- INFO: epoch: 708, batch: 1, loss: 0.7337482571601868, modularity_loss: -0.09345141798257828, purity_loss: 0.7563894987106323, regularization_loss: 0.07081017643213272
16:10:10 --- INFO: epoch: 709, batch: 1, loss: 0.733674168586731, modularity_loss: -0.09344912320375443, purity_loss: 0.7563125491142273, regularization_loss: 0.07081073522567749
16:10:10 --- INFO: epoch: 710, batch: 1, loss: 0.7336006164550781, modularity_loss: -0.09344764798879623, purity_loss: 0.7562371492385864, regularization_loss: 0.07081113010644913
16:10:10 --- INFO: epoch: 711, batch: 1, loss: 0.7335270047187805, modularity_loss: -0.09344717115163803, purity_loss: 0.7561629414558411, regularization_loss: 0.07081124931573868
16:10:10 --- INFO: epoch: 712, batch: 1, loss: 0.7334530353546143, modularity_loss: -0.0934462770819664, purity_loss: 0.7560878992080688, regularization_loss: 0.0708114355802536
16:10:10 --- INFO: epoch: 713, batch: 1, loss: 0.7333797216415405, modularity_loss: -0.09344512224197388, purity_loss: 0.7560132145881653, regularization_loss: 0.07081165164709091
16:10:10 --- INFO: epoch: 714, batch: 1, loss: 0.7333059310913086, modularity_loss: -0.09344514459371567, purity_loss: 0.7559396028518677, regularization_loss: 0.07081149518489838
16:10:10 --- INFO: epoch: 715, batch: 1, loss: 0.7332320213317871, modularity_loss: -0.09344608336687088, purity_loss: 0.755867063999176, regularization_loss: 0.07081106305122375
16:10:10 --- INFO: epoch: 716, batch: 1, loss: 0.7331584095954895, modularity_loss: -0.09344623237848282, purity_loss: 0.7557938098907471, regularization_loss: 0.07081081718206406
16:10:10 --- INFO: epoch: 717, batch: 1, loss: 0.7330847382545471, modularity_loss: -0.09344679862260818, purity_loss: 0.7557210922241211, regularization_loss: 0.07081042230129242
16:10:10 --- INFO: epoch: 718, batch: 1, loss: 0.7330108284950256, modularity_loss: -0.09344621747732162, purity_loss: 0.7556467056274414, regularization_loss: 0.07081036269664764
16:10:10 --- INFO: epoch: 719, batch: 1, loss: 0.732937216758728, modularity_loss: -0.09344596415758133, purity_loss: 0.7555729150772095, regularization_loss: 0.0708102434873581
16:10:10 --- INFO: epoch: 720, batch: 1, loss: 0.7328634858131409, modularity_loss: -0.0934460312128067, purity_loss: 0.7554993629455566, regularization_loss: 0.07081014662981033
16:10:10 --- INFO: epoch: 721, batch: 1, loss: 0.7327899932861328, modularity_loss: -0.09344622492790222, purity_loss: 0.7554261684417725, regularization_loss: 0.07081003487110138
16:10:10 --- INFO: epoch: 722, batch: 1, loss: 0.7327159643173218, modularity_loss: -0.09344513714313507, purity_loss: 0.7553507685661316, regularization_loss: 0.07081030309200287
16:10:10 --- INFO: epoch: 723, batch: 1, loss: 0.7326423525810242, modularity_loss: -0.09344430267810822, purity_loss: 0.7552762031555176, regularization_loss: 0.070810467004776
16:10:10 --- INFO: epoch: 724, batch: 1, loss: 0.7325683832168579, modularity_loss: -0.0934438481926918, purity_loss: 0.7552016973495483, regularization_loss: 0.07081050425767899
16:10:10 --- INFO: epoch: 725, batch: 1, loss: 0.7324947118759155, modularity_loss: -0.09344375878572464, purity_loss: 0.7551280856132507, regularization_loss: 0.07081039994955063
16:10:10 --- INFO: epoch: 726, batch: 1, loss: 0.7324206829071045, modularity_loss: -0.09344261884689331, purity_loss: 0.7550527453422546, regularization_loss: 0.07081054896116257
16:10:10 --- INFO: epoch: 727, batch: 1, loss: 0.7323469519615173, modularity_loss: -0.09344081580638885, purity_loss: 0.7549768686294556, regularization_loss: 0.07081091403961182
16:10:10 --- INFO: epoch: 728, batch: 1, loss: 0.7322731018066406, modularity_loss: -0.09344013035297394, purity_loss: 0.754902184009552, regularization_loss: 0.07081106305122375
16:10:10 --- INFO: epoch: 729, batch: 1, loss: 0.7321990132331848, modularity_loss: -0.09344057738780975, purity_loss: 0.7548286318778992, regularization_loss: 0.0708109587430954
16:10:10 --- INFO: epoch: 730, batch: 1, loss: 0.7321247458457947, modularity_loss: -0.09344179928302765, purity_loss: 0.754755973815918, regularization_loss: 0.07081058621406555
16:10:10 --- INFO: epoch: 731, batch: 1, loss: 0.7320511341094971, modularity_loss: -0.09344331920146942, purity_loss: 0.7546844482421875, regularization_loss: 0.07081001251935959
16:10:10 --- INFO: epoch: 732, batch: 1, loss: 0.7319771647453308, modularity_loss: -0.09344324469566345, purity_loss: 0.7546106576919556, regularization_loss: 0.0708097442984581
16:10:10 --- INFO: epoch: 733, batch: 1, loss: 0.7319025993347168, modularity_loss: -0.09344175457954407, purity_loss: 0.7545344829559326, regularization_loss: 0.07080984115600586
16:10:10 --- INFO: epoch: 734, batch: 1, loss: 0.7318286299705505, modularity_loss: -0.0934395045042038, purity_loss: 0.7544578313827515, regularization_loss: 0.07081032544374466
16:10:10 --- INFO: epoch: 735, batch: 1, loss: 0.7317546606063843, modularity_loss: -0.09343845397233963, purity_loss: 0.7543823719024658, regularization_loss: 0.0708107203245163
16:10:10 --- INFO: epoch: 736, batch: 1, loss: 0.7316802740097046, modularity_loss: -0.0934387668967247, purity_loss: 0.7543081641197205, regularization_loss: 0.07081089913845062
16:10:10 --- INFO: epoch: 737, batch: 1, loss: 0.7316058874130249, modularity_loss: -0.09344007074832916, purity_loss: 0.7542352080345154, regularization_loss: 0.07081073522567749
16:10:10 --- INFO: epoch: 738, batch: 1, loss: 0.7315313816070557, modularity_loss: -0.09344176203012466, purity_loss: 0.7541629076004028, regularization_loss: 0.0708102211356163
16:10:10 --- INFO: epoch: 739, batch: 1, loss: 0.7314568758010864, modularity_loss: -0.09344326704740524, purity_loss: 0.7540906071662903, regularization_loss: 0.07080952823162079
16:10:10 --- INFO: epoch: 740, batch: 1, loss: 0.7313823699951172, modularity_loss: -0.09344286471605301, purity_loss: 0.7540159821510315, regularization_loss: 0.0708092600107193
16:10:10 --- INFO: epoch: 741, batch: 1, loss: 0.731307864189148, modularity_loss: -0.09344221651554108, purity_loss: 0.7539408802986145, regularization_loss: 0.07080919295549393
16:10:10 --- INFO: epoch: 742, batch: 1, loss: 0.7312331795692444, modularity_loss: -0.09344156086444855, purity_loss: 0.7538653612136841, regularization_loss: 0.07080937176942825
16:10:10 --- INFO: epoch: 743, batch: 1, loss: 0.7311588525772095, modularity_loss: -0.09343986213207245, purity_loss: 0.7537885308265686, regularization_loss: 0.07081016153097153
16:10:10 --- INFO: epoch: 744, batch: 1, loss: 0.731084406375885, modularity_loss: -0.09343889355659485, purity_loss: 0.7537124752998352, regularization_loss: 0.07081083208322525
16:10:10 --- INFO: epoch: 745, batch: 1, loss: 0.731009840965271, modularity_loss: -0.09343871474266052, purity_loss: 0.753637433052063, regularization_loss: 0.07081110030412674
16:10:10 --- INFO: epoch: 746, batch: 1, loss: 0.7309349775314331, modularity_loss: -0.09343912452459335, purity_loss: 0.7535632252693176, regularization_loss: 0.07081086188554764
16:10:10 --- INFO: epoch: 747, batch: 1, loss: 0.7308602333068848, modularity_loss: -0.09343983232975006, purity_loss: 0.7534898519515991, regularization_loss: 0.0708102136850357
16:10:10 --- INFO: epoch: 748, batch: 1, loss: 0.7307854294776917, modularity_loss: -0.09343921393156052, purity_loss: 0.7534148693084717, regularization_loss: 0.07080977410078049
16:10:10 --- INFO: epoch: 749, batch: 1, loss: 0.7307106852531433, modularity_loss: -0.09343762695789337, purity_loss: 0.7533385753631592, regularization_loss: 0.07080972194671631
16:10:10 --- INFO: epoch: 750, batch: 1, loss: 0.7306358218193054, modularity_loss: -0.09343694150447845, purity_loss: 0.7532630562782288, regularization_loss: 0.07080968469381332
16:10:10 --- INFO: epoch: 751, batch: 1, loss: 0.730560839176178, modularity_loss: -0.09343720227479935, purity_loss: 0.75318843126297, regularization_loss: 0.07080961018800735
16:10:10 --- INFO: epoch: 752, batch: 1, loss: 0.7304856777191162, modularity_loss: -0.09343820810317993, purity_loss: 0.7531144618988037, regularization_loss: 0.07080941647291183
16:10:10 --- INFO: epoch: 753, batch: 1, loss: 0.7304109334945679, modularity_loss: -0.09343943744897842, purity_loss: 0.7530412673950195, regularization_loss: 0.07080907374620438
16:10:10 --- INFO: epoch: 754, batch: 1, loss: 0.7303355932235718, modularity_loss: -0.09343910217285156, purity_loss: 0.7529656887054443, regularization_loss: 0.07080903649330139
16:10:10 --- INFO: epoch: 755, batch: 1, loss: 0.7302603125572205, modularity_loss: -0.09343743324279785, purity_loss: 0.7528883814811707, regularization_loss: 0.07080936431884766
16:10:10 --- INFO: epoch: 756, batch: 1, loss: 0.7301852107048035, modularity_loss: -0.09343624114990234, purity_loss: 0.7528119087219238, regularization_loss: 0.07080952078104019
16:10:10 --- INFO: epoch: 757, batch: 1, loss: 0.7301098108291626, modularity_loss: -0.09343577921390533, purity_loss: 0.7527360320091248, regularization_loss: 0.07080954313278198
16:10:10 --- INFO: epoch: 758, batch: 1, loss: 0.7300344109535217, modularity_loss: -0.09343615919351578, purity_loss: 0.7526610493659973, regularization_loss: 0.07080952078104019
16:10:10 --- INFO: epoch: 759, batch: 1, loss: 0.7299590110778809, modularity_loss: -0.09343709796667099, purity_loss: 0.7525866627693176, regularization_loss: 0.07080941647291183
16:10:10 --- INFO: epoch: 760, batch: 1, loss: 0.7298835515975952, modularity_loss: -0.09343817085027695, purity_loss: 0.752512514591217, regularization_loss: 0.07080921530723572
16:10:10 --- INFO: epoch: 761, batch: 1, loss: 0.7298078536987305, modularity_loss: -0.09343765676021576, purity_loss: 0.7524361610412598, regularization_loss: 0.07080936431884766
16:10:10 --- INFO: epoch: 762, batch: 1, loss: 0.7297323942184448, modularity_loss: -0.09343593567609787, purity_loss: 0.7523584365844727, regularization_loss: 0.07080987095832825
16:10:10 --- INFO: epoch: 763, batch: 1, loss: 0.7296568155288696, modularity_loss: -0.0934349000453949, purity_loss: 0.7522816061973572, regularization_loss: 0.07081010192632675
16:10:10 --- INFO: epoch: 764, batch: 1, loss: 0.7295811176300049, modularity_loss: -0.09343492984771729, purity_loss: 0.7522059082984924, regularization_loss: 0.07081015408039093
16:10:10 --- INFO: epoch: 765, batch: 1, loss: 0.7295050024986267, modularity_loss: -0.09343577921390533, purity_loss: 0.7521309852600098, regularization_loss: 0.07080980390310287
16:10:10 --- INFO: epoch: 766, batch: 1, loss: 0.7294290065765381, modularity_loss: -0.09343720227479935, purity_loss: 0.7520569562911987, regularization_loss: 0.0708092600107193
16:10:10 --- INFO: epoch: 767, batch: 1, loss: 0.7293529510498047, modularity_loss: -0.0934387668967247, purity_loss: 0.7519830465316772, regularization_loss: 0.07080864906311035
16:10:10 --- INFO: epoch: 768, batch: 1, loss: 0.7292770743370056, modularity_loss: -0.0934399738907814, purity_loss: 0.7519088983535767, regularization_loss: 0.07080817222595215
16:10:10 --- INFO: epoch: 769, batch: 1, loss: 0.7292007207870483, modularity_loss: -0.09343923628330231, purity_loss: 0.751831591129303, regularization_loss: 0.07080838829278946
16:10:10 --- INFO: epoch: 770, batch: 1, loss: 0.7291245460510254, modularity_loss: -0.09343841671943665, purity_loss: 0.7517541646957397, regularization_loss: 0.07080880552530289
16:10:10 --- INFO: epoch: 771, batch: 1, loss: 0.7290480732917786, modularity_loss: -0.09343776851892471, purity_loss: 0.751676619052887, regularization_loss: 0.07080923020839691
16:10:10 --- INFO: epoch: 772, batch: 1, loss: 0.7289718389511108, modularity_loss: -0.09343750029802322, purity_loss: 0.751599907875061, regularization_loss: 0.07080943137407303
16:10:10 --- INFO: epoch: 773, batch: 1, loss: 0.7288953065872192, modularity_loss: -0.09343758970499039, purity_loss: 0.7515236139297485, regularization_loss: 0.0708092600107193
16:10:10 --- INFO: epoch: 774, batch: 1, loss: 0.728818953037262, modularity_loss: -0.09343800693750381, purity_loss: 0.7514481544494629, regularization_loss: 0.07080880552530289
16:10:10 --- INFO: epoch: 775, batch: 1, loss: 0.7287421226501465, modularity_loss: -0.0934372991323471, purity_loss: 0.7513707876205444, regularization_loss: 0.07080866396427155
16:10:10 --- INFO: epoch: 776, batch: 1, loss: 0.7286655902862549, modularity_loss: -0.09343595802783966, purity_loss: 0.7512925863265991, regularization_loss: 0.07080896943807602
16:10:10 --- INFO: epoch: 777, batch: 1, loss: 0.7285888195037842, modularity_loss: -0.09343592822551727, purity_loss: 0.7512155175209045, regularization_loss: 0.0708092600107193
16:10:10 --- INFO: epoch: 778, batch: 1, loss: 0.7285117506980896, modularity_loss: -0.09343715757131577, purity_loss: 0.7511395215988159, regularization_loss: 0.07080937176942825
16:10:10 --- INFO: epoch: 779, batch: 1, loss: 0.7284345626831055, modularity_loss: -0.09343919157981873, purity_loss: 0.7510645389556885, regularization_loss: 0.07080920040607452
16:10:10 --- INFO: epoch: 780, batch: 1, loss: 0.7283572554588318, modularity_loss: -0.09344140440225601, purity_loss: 0.7509899735450745, regularization_loss: 0.07080869376659393
16:10:11 --- INFO: epoch: 781, batch: 1, loss: 0.7282804250717163, modularity_loss: -0.09344327449798584, purity_loss: 0.7509157061576843, regularization_loss: 0.07080799341201782
16:10:11 --- INFO: epoch: 782, batch: 1, loss: 0.7282027006149292, modularity_loss: -0.09344305098056793, purity_loss: 0.7508379220962524, regularization_loss: 0.0708078145980835
16:10:11 --- INFO: epoch: 783, batch: 1, loss: 0.7281250953674316, modularity_loss: -0.09344135224819183, purity_loss: 0.750758171081543, regularization_loss: 0.0708082988858223
16:10:11 --- INFO: epoch: 784, batch: 1, loss: 0.7280476689338684, modularity_loss: -0.09344049543142319, purity_loss: 0.7506791949272156, regularization_loss: 0.07080896198749542
16:10:11 --- INFO: epoch: 785, batch: 1, loss: 0.7279700040817261, modularity_loss: -0.09344081580638885, purity_loss: 0.7506012916564941, regularization_loss: 0.07080954313278198
16:10:11 --- INFO: epoch: 786, batch: 1, loss: 0.7278921008110046, modularity_loss: -0.09344218671321869, purity_loss: 0.7505245208740234, regularization_loss: 0.0708097517490387
16:10:11 --- INFO: epoch: 787, batch: 1, loss: 0.7278141975402832, modularity_loss: -0.09344418346881866, purity_loss: 0.7504488825798035, regularization_loss: 0.07080947607755661
16:10:11 --- INFO: epoch: 788, batch: 1, loss: 0.7277359962463379, modularity_loss: -0.09344631433486938, purity_loss: 0.7503734827041626, regularization_loss: 0.07080881297588348
16:10:11 --- INFO: epoch: 789, batch: 1, loss: 0.7276578545570374, modularity_loss: -0.09344828873872757, purity_loss: 0.7502979636192322, regularization_loss: 0.07080816477537155
16:10:11 --- INFO: epoch: 790, batch: 1, loss: 0.7275795936584473, modularity_loss: -0.09344980865716934, purity_loss: 0.7502216100692749, regularization_loss: 0.0708077996969223
16:10:11 --- INFO: epoch: 791, batch: 1, loss: 0.7275009751319885, modularity_loss: -0.09345079213380814, purity_loss: 0.7501439452171326, regularization_loss: 0.07080784440040588
16:10:11 --- INFO: epoch: 792, batch: 1, loss: 0.7274224758148193, modularity_loss: -0.09345121681690216, purity_loss: 0.7500655055046082, regularization_loss: 0.07080817222595215
16:10:11 --- INFO: epoch: 793, batch: 1, loss: 0.7273439168930054, modularity_loss: -0.09345122426748276, purity_loss: 0.7499865889549255, regularization_loss: 0.07080855965614319
16:10:11 --- INFO: epoch: 794, batch: 1, loss: 0.7272650003433228, modularity_loss: -0.09345106780529022, purity_loss: 0.7499071955680847, regularization_loss: 0.07080884277820587
16:10:11 --- INFO: epoch: 795, batch: 1, loss: 0.7271860241889954, modularity_loss: -0.09345097094774246, purity_loss: 0.749828040599823, regularization_loss: 0.07080896198749542
16:10:11 --- INFO: epoch: 796, batch: 1, loss: 0.7271069884300232, modularity_loss: -0.09345122426748276, purity_loss: 0.7497492432594299, regularization_loss: 0.07080896943807602
16:10:11 --- INFO: epoch: 797, batch: 1, loss: 0.7270282506942749, modularity_loss: -0.0934518501162529, purity_loss: 0.7496711611747742, regularization_loss: 0.07080892473459244
16:10:11 --- INFO: epoch: 798, batch: 1, loss: 0.7269487977027893, modularity_loss: -0.09345142543315887, purity_loss: 0.7495909929275513, regularization_loss: 0.07080922275781631
16:10:11 --- INFO: epoch: 799, batch: 1, loss: 0.7268697023391724, modularity_loss: -0.09345186501741409, purity_loss: 0.749512255191803, regularization_loss: 0.07080933451652527
16:10:11 --- INFO: epoch: 800, batch: 1, loss: 0.726790189743042, modularity_loss: -0.09345321357250214, purity_loss: 0.749434232711792, regularization_loss: 0.07080918550491333
16:10:11 --- INFO: epoch: 801, batch: 1, loss: 0.7267107367515564, modularity_loss: -0.09345534443855286, purity_loss: 0.749357283115387, regularization_loss: 0.07080882042646408
16:10:11 --- INFO: epoch: 802, batch: 1, loss: 0.7266309261322021, modularity_loss: -0.09345720708370209, purity_loss: 0.7492796182632446, regularization_loss: 0.0708085149526596
16:10:11 --- INFO: epoch: 803, batch: 1, loss: 0.7265508770942688, modularity_loss: -0.0934586226940155, purity_loss: 0.7492011189460754, regularization_loss: 0.07080838829278946
16:10:11 --- INFO: epoch: 804, batch: 1, loss: 0.7264704704284668, modularity_loss: -0.0934595987200737, purity_loss: 0.7491216063499451, regularization_loss: 0.07080848515033722
16:10:11 --- INFO: epoch: 805, batch: 1, loss: 0.7263906002044678, modularity_loss: -0.09346017241477966, purity_loss: 0.7490419745445251, regularization_loss: 0.0708087906241417
16:10:11 --- INFO: epoch: 806, batch: 1, loss: 0.726310133934021, modularity_loss: -0.09346190840005875, purity_loss: 0.7489632964134216, regularization_loss: 0.07080873847007751
16:10:11 --- INFO: epoch: 807, batch: 1, loss: 0.726229190826416, modularity_loss: -0.09346440434455872, purity_loss: 0.7488852739334106, regularization_loss: 0.07080831378698349
16:10:11 --- INFO: epoch: 808, batch: 1, loss: 0.7261483669281006, modularity_loss: -0.0934656411409378, purity_loss: 0.7488059401512146, regularization_loss: 0.07080806791782379
16:10:11 --- INFO: epoch: 809, batch: 1, loss: 0.726067304611206, modularity_loss: -0.09346573054790497, purity_loss: 0.7487248778343201, regularization_loss: 0.07080814987421036
16:10:11 --- INFO: epoch: 810, batch: 1, loss: 0.7259861826896667, modularity_loss: -0.09346655756235123, purity_loss: 0.7486445307731628, regularization_loss: 0.07080823183059692
16:10:11 --- INFO: epoch: 811, batch: 1, loss: 0.7259049415588379, modularity_loss: -0.09346816688776016, purity_loss: 0.7485647797584534, regularization_loss: 0.07080833613872528
16:10:11 --- INFO: epoch: 812, batch: 1, loss: 0.725823700428009, modularity_loss: -0.0934688001871109, purity_loss: 0.7484837174415588, regularization_loss: 0.07080880552530289
16:10:11 --- INFO: epoch: 813, batch: 1, loss: 0.7257425785064697, modularity_loss: -0.09346867352724075, purity_loss: 0.7484018802642822, regularization_loss: 0.07080939412117004
16:10:11 --- INFO: epoch: 814, batch: 1, loss: 0.7256608009338379, modularity_loss: -0.09347082674503326, purity_loss: 0.7483224868774414, regularization_loss: 0.07080914825201035
16:10:11 --- INFO: epoch: 815, batch: 1, loss: 0.7255795001983643, modularity_loss: -0.09347321838140488, purity_loss: 0.7482441067695618, regularization_loss: 0.07080863416194916
16:10:11 --- INFO: epoch: 816, batch: 1, loss: 0.7254977822303772, modularity_loss: -0.09347404539585114, purity_loss: 0.7481634616851807, regularization_loss: 0.07080838829278946
16:10:11 --- INFO: epoch: 817, batch: 1, loss: 0.725415825843811, modularity_loss: -0.09347359091043472, purity_loss: 0.748080849647522, regularization_loss: 0.07080855965614319
16:10:11 --- INFO: epoch: 818, batch: 1, loss: 0.725334107875824, modularity_loss: -0.09347376972436905, purity_loss: 0.7479991316795349, regularization_loss: 0.0708087608218193
16:10:11 --- INFO: epoch: 819, batch: 1, loss: 0.7252522110939026, modularity_loss: -0.09347479790449142, purity_loss: 0.7479181289672852, regularization_loss: 0.07080889493227005
16:10:11 --- INFO: epoch: 820, batch: 1, loss: 0.7251698970794678, modularity_loss: -0.09347660839557648, purity_loss: 0.7478377223014832, regularization_loss: 0.07080881297588348
16:10:11 --- INFO: epoch: 821, batch: 1, loss: 0.7250877618789673, modularity_loss: -0.09347885847091675, purity_loss: 0.747758150100708, regularization_loss: 0.07080849260091782
16:10:11 --- INFO: epoch: 822, batch: 1, loss: 0.7250051498413086, modularity_loss: -0.09347987174987793, purity_loss: 0.7476766109466553, regularization_loss: 0.07080841064453125
16:10:11 --- INFO: epoch: 823, batch: 1, loss: 0.7249224185943604, modularity_loss: -0.09347989410161972, purity_loss: 0.7475937008857727, regularization_loss: 0.07080861181020737
16:10:11 --- INFO: epoch: 824, batch: 1, loss: 0.7248393893241882, modularity_loss: -0.09348077327013016, purity_loss: 0.7475114464759827, regularization_loss: 0.07080871611833572
16:10:11 --- INFO: epoch: 825, batch: 1, loss: 0.724756121635437, modularity_loss: -0.09348253905773163, purity_loss: 0.7474300265312195, regularization_loss: 0.07080866396427155
16:10:11 --- INFO: epoch: 826, batch: 1, loss: 0.7246728539466858, modularity_loss: -0.09348495304584503, purity_loss: 0.7473493218421936, regularization_loss: 0.07080849260091782
16:10:11 --- INFO: epoch: 827, batch: 1, loss: 0.7245895862579346, modularity_loss: -0.09348761290311813, purity_loss: 0.7472689747810364, regularization_loss: 0.07080820202827454
16:10:11 --- INFO: epoch: 828, batch: 1, loss: 0.7245057225227356, modularity_loss: -0.09348870813846588, purity_loss: 0.7471861839294434, regularization_loss: 0.07080824673175812
16:10:11 --- INFO: epoch: 829, batch: 1, loss: 0.7244219779968262, modularity_loss: -0.09348990023136139, purity_loss: 0.7471036314964294, regularization_loss: 0.07080826163291931
16:10:11 --- INFO: epoch: 830, batch: 1, loss: 0.7243378162384033, modularity_loss: -0.09349124878644943, purity_loss: 0.7470207214355469, regularization_loss: 0.07080832868814468
16:10:11 --- INFO: epoch: 831, batch: 1, loss: 0.7242535352706909, modularity_loss: -0.09349275380373001, purity_loss: 0.7469378709793091, regularization_loss: 0.07080838829278946
16:10:11 --- INFO: epoch: 832, batch: 1, loss: 0.7241690754890442, modularity_loss: -0.09349314868450165, purity_loss: 0.7468534708023071, regularization_loss: 0.0708087608218193
16:10:11 --- INFO: epoch: 833, batch: 1, loss: 0.7240844368934631, modularity_loss: -0.09349419176578522, purity_loss: 0.7467696666717529, regularization_loss: 0.07080896198749542
16:10:11 --- INFO: epoch: 834, batch: 1, loss: 0.7239996194839478, modularity_loss: -0.09349607676267624, purity_loss: 0.7466866970062256, regularization_loss: 0.07080899178981781
16:10:11 --- INFO: epoch: 835, batch: 1, loss: 0.723914623260498, modularity_loss: -0.0934986099600792, purity_loss: 0.746604323387146, regularization_loss: 0.07080889493227005
16:10:11 --- INFO: epoch: 836, batch: 1, loss: 0.7238292694091797, modularity_loss: -0.093501515686512, purity_loss: 0.7465220093727112, regularization_loss: 0.07080880552530289
16:10:11 --- INFO: epoch: 837, batch: 1, loss: 0.7237440347671509, modularity_loss: -0.09350433945655823, purity_loss: 0.7464396357536316, regularization_loss: 0.07080873847007751
16:10:11 --- INFO: epoch: 838, batch: 1, loss: 0.7236579656600952, modularity_loss: -0.0935053825378418, purity_loss: 0.7463542819023132, regularization_loss: 0.07080907374620438
16:10:11 --- INFO: epoch: 839, batch: 1, loss: 0.7235727310180664, modularity_loss: -0.09350528568029404, purity_loss: 0.7462682127952576, regularization_loss: 0.07080980390310287
16:10:11 --- INFO: epoch: 840, batch: 1, loss: 0.7234877347946167, modularity_loss: -0.09350600093603134, purity_loss: 0.7461835145950317, regularization_loss: 0.0708102136850357
16:10:11 --- INFO: epoch: 841, batch: 1, loss: 0.7234022617340088, modularity_loss: -0.09350785613059998, purity_loss: 0.7460999488830566, regularization_loss: 0.07081018388271332
16:10:11 --- INFO: epoch: 842, batch: 1, loss: 0.7233165502548218, modularity_loss: -0.09351076185703278, purity_loss: 0.746017575263977, regularization_loss: 0.0708097517490387
16:10:11 --- INFO: epoch: 843, batch: 1, loss: 0.7232310771942139, modularity_loss: -0.09351421147584915, purity_loss: 0.7459361553192139, regularization_loss: 0.07080910354852676
16:10:11 --- INFO: epoch: 844, batch: 1, loss: 0.7231454253196716, modularity_loss: -0.09351767599582672, purity_loss: 0.7458546161651611, regularization_loss: 0.07080847024917603
16:10:11 --- INFO: epoch: 845, batch: 1, loss: 0.7230596542358398, modularity_loss: -0.0935206487774849, purity_loss: 0.7457722425460815, regularization_loss: 0.0708080381155014
16:10:11 --- INFO: epoch: 846, batch: 1, loss: 0.7229737043380737, modularity_loss: -0.0935230404138565, purity_loss: 0.7456886768341064, regularization_loss: 0.0708080381155014
16:10:11 --- INFO: epoch: 847, batch: 1, loss: 0.7228877544403076, modularity_loss: -0.09352477639913559, purity_loss: 0.7456040978431702, regularization_loss: 0.07080840319395065
16:10:11 --- INFO: epoch: 848, batch: 1, loss: 0.7228016257286072, modularity_loss: -0.09352600574493408, purity_loss: 0.745518684387207, regularization_loss: 0.07080893963575363
16:10:11 --- INFO: epoch: 849, batch: 1, loss: 0.722715437412262, modularity_loss: -0.0935269445180893, purity_loss: 0.7454329133033752, regularization_loss: 0.07080946117639542
16:10:11 --- INFO: epoch: 850, batch: 1, loss: 0.722629189491272, modularity_loss: -0.09352781623601913, purity_loss: 0.745347261428833, regularization_loss: 0.07080977410078049
16:10:11 --- INFO: epoch: 851, batch: 1, loss: 0.7225428223609924, modularity_loss: -0.09352883696556091, purity_loss: 0.7452617883682251, regularization_loss: 0.07080988585948944
16:10:11 --- INFO: epoch: 852, batch: 1, loss: 0.7224564552307129, modularity_loss: -0.09353011846542358, purity_loss: 0.7451767325401306, regularization_loss: 0.07080983370542526
16:10:11 --- INFO: epoch: 853, batch: 1, loss: 0.722370445728302, modularity_loss: -0.09353169053792953, purity_loss: 0.7450923919677734, regularization_loss: 0.07080972194671631
16:10:11 --- INFO: epoch: 854, batch: 1, loss: 0.7222837805747986, modularity_loss: -0.09353213012218475, purity_loss: 0.7450059056282043, regularization_loss: 0.0708099976181984
16:10:11 --- INFO: epoch: 855, batch: 1, loss: 0.7221980094909668, modularity_loss: -0.09353209286928177, purity_loss: 0.74491947889328, regularization_loss: 0.07081063836812973
16:10:11 --- INFO: epoch: 856, batch: 1, loss: 0.7221117615699768, modularity_loss: -0.09353455901145935, purity_loss: 0.7448355555534363, regularization_loss: 0.07081077247858047
16:10:11 --- INFO: epoch: 857, batch: 1, loss: 0.7220246195793152, modularity_loss: -0.09353911876678467, purity_loss: 0.744753360748291, regularization_loss: 0.07081039249897003
16:10:11 --- INFO: epoch: 858, batch: 1, loss: 0.7219386696815491, modularity_loss: -0.09354466944932938, purity_loss: 0.7446737289428711, regularization_loss: 0.07080962508916855
16:10:11 --- INFO: epoch: 859, batch: 1, loss: 0.7218523025512695, modularity_loss: -0.09354755282402039, purity_loss: 0.7445905208587646, regularization_loss: 0.07080931216478348
16:10:11 --- INFO: epoch: 860, batch: 1, loss: 0.7217655777931213, modularity_loss: -0.09354859590530396, purity_loss: 0.7445048689842224, regularization_loss: 0.07080931216478348
16:10:11 --- INFO: epoch: 861, batch: 1, loss: 0.7216784954071045, modularity_loss: -0.09354833513498306, purity_loss: 0.7444171905517578, regularization_loss: 0.07080966979265213
16:10:11 --- INFO: epoch: 862, batch: 1, loss: 0.7215926051139832, modularity_loss: -0.09354770928621292, purity_loss: 0.7443299889564514, regularization_loss: 0.07081031054258347
16:10:11 --- INFO: epoch: 863, batch: 1, loss: 0.7215062379837036, modularity_loss: -0.093548983335495, purity_loss: 0.7442445755004883, regularization_loss: 0.07081065326929092
16:10:11 --- INFO: epoch: 864, batch: 1, loss: 0.721419095993042, modularity_loss: -0.09355209767818451, purity_loss: 0.7441606521606445, regularization_loss: 0.07081054896116257
16:10:11 --- INFO: epoch: 865, batch: 1, loss: 0.7213319540023804, modularity_loss: -0.09355635941028595, purity_loss: 0.7440782785415649, regularization_loss: 0.07081001996994019
16:10:11 --- INFO: epoch: 866, batch: 1, loss: 0.7212458848953247, modularity_loss: -0.09356074780225754, purity_loss: 0.74399733543396, regularization_loss: 0.07080932706594467
16:10:11 --- INFO: epoch: 867, batch: 1, loss: 0.7211595773696899, modularity_loss: -0.09356296062469482, purity_loss: 0.7439134120941162, regularization_loss: 0.07080911844968796
16:10:11 --- INFO: epoch: 868, batch: 1, loss: 0.7210725545883179, modularity_loss: -0.09356313198804855, purity_loss: 0.743826150894165, regularization_loss: 0.07080952823162079
16:10:11 --- INFO: epoch: 869, batch: 1, loss: 0.7209854125976562, modularity_loss: -0.09356208890676498, purity_loss: 0.7437371015548706, regularization_loss: 0.07081039994955063
16:10:11 --- INFO: epoch: 870, batch: 1, loss: 0.7208990454673767, modularity_loss: -0.09356105327606201, purity_loss: 0.7436486482620239, regularization_loss: 0.0708114355802536
16:10:11 --- INFO: epoch: 871, batch: 1, loss: 0.7208124995231628, modularity_loss: -0.09356240928173065, purity_loss: 0.7435630559921265, regularization_loss: 0.07081187516450882
16:10:11 --- INFO: epoch: 872, batch: 1, loss: 0.7207251787185669, modularity_loss: -0.09356599301099777, purity_loss: 0.7434795498847961, regularization_loss: 0.07081159949302673
16:10:11 --- INFO: epoch: 873, batch: 1, loss: 0.72063809633255, modularity_loss: -0.09357094019651413, purity_loss: 0.7433982491493225, regularization_loss: 0.07081077992916107
16:10:11 --- INFO: epoch: 874, batch: 1, loss: 0.7205514311790466, modularity_loss: -0.09357453882694244, purity_loss: 0.7433158159255981, regularization_loss: 0.07081017643213272
16:10:11 --- INFO: epoch: 875, batch: 1, loss: 0.7204645872116089, modularity_loss: -0.09357640892267227, purity_loss: 0.743230938911438, regularization_loss: 0.07081006467342377
16:10:11 --- INFO: epoch: 876, batch: 1, loss: 0.7203776240348816, modularity_loss: -0.09357686340808868, purity_loss: 0.7431440353393555, regularization_loss: 0.07081045210361481
16:10:11 --- INFO: epoch: 877, batch: 1, loss: 0.7202910780906677, modularity_loss: -0.09357821941375732, purity_loss: 0.743058443069458, regularization_loss: 0.07081083208322525
16:10:11 --- INFO: epoch: 878, batch: 1, loss: 0.7202047109603882, modularity_loss: -0.09358061105012894, purity_loss: 0.7429743409156799, regularization_loss: 0.07081099599599838
16:10:11 --- INFO: epoch: 879, batch: 1, loss: 0.7201183438301086, modularity_loss: -0.09358387440443039, purity_loss: 0.7428913712501526, regularization_loss: 0.07081086188554764
16:10:11 --- INFO: epoch: 880, batch: 1, loss: 0.7200325727462769, modularity_loss: -0.09358743578195572, purity_loss: 0.7428094744682312, regularization_loss: 0.07081050425767899
16:10:11 --- INFO: epoch: 881, batch: 1, loss: 0.7199464440345764, modularity_loss: -0.09358913451433182, purity_loss: 0.742725133895874, regularization_loss: 0.07081044465303421
16:10:11 --- INFO: epoch: 882, batch: 1, loss: 0.7198601365089417, modularity_loss: -0.09358923137187958, purity_loss: 0.7426386475563049, regularization_loss: 0.07081073522567749
16:10:11 --- INFO: epoch: 883, batch: 1, loss: 0.7197743654251099, modularity_loss: -0.09359009563922882, purity_loss: 0.7425535917282104, regularization_loss: 0.07081088423728943
16:10:11 --- INFO: epoch: 884, batch: 1, loss: 0.7196884751319885, modularity_loss: -0.09359212964773178, purity_loss: 0.7424697279930115, regularization_loss: 0.07081086188554764
16:10:11 --- INFO: epoch: 885, batch: 1, loss: 0.7196028828620911, modularity_loss: -0.09359530359506607, purity_loss: 0.7423875331878662, regularization_loss: 0.07081063836812973
16:10:11 --- INFO: epoch: 886, batch: 1, loss: 0.7195171117782593, modularity_loss: -0.09359767287969589, purity_loss: 0.7423040866851807, regularization_loss: 0.07081068307161331
16:10:12 --- INFO: epoch: 887, batch: 1, loss: 0.7194314002990723, modularity_loss: -0.09359930455684662, purity_loss: 0.7422197461128235, regularization_loss: 0.0708109587430954
16:10:12 --- INFO: epoch: 888, batch: 1, loss: 0.7193460464477539, modularity_loss: -0.09360203146934509, purity_loss: 0.7421370148658752, regularization_loss: 0.07081104815006256
16:10:12 --- INFO: epoch: 889, batch: 1, loss: 0.7192603945732117, modularity_loss: -0.0936056524515152, purity_loss: 0.7420551180839539, regularization_loss: 0.07081093639135361
16:10:12 --- INFO: epoch: 890, batch: 1, loss: 0.7191751003265381, modularity_loss: -0.09360804408788681, purity_loss: 0.7419721484184265, regularization_loss: 0.07081098109483719
16:10:12 --- INFO: epoch: 891, batch: 1, loss: 0.7190898656845093, modularity_loss: -0.09361089766025543, purity_loss: 0.7418899536132812, regularization_loss: 0.07081079483032227
16:10:12 --- INFO: epoch: 892, batch: 1, loss: 0.7190046906471252, modularity_loss: -0.09361401200294495, purity_loss: 0.7418082356452942, regularization_loss: 0.07081048935651779
16:10:12 --- INFO: epoch: 893, batch: 1, loss: 0.7189193367958069, modularity_loss: -0.09361562132835388, purity_loss: 0.7417242527008057, regularization_loss: 0.07081069052219391
16:10:12 --- INFO: epoch: 894, batch: 1, loss: 0.7188342213630676, modularity_loss: -0.09361779689788818, purity_loss: 0.7416410446166992, regularization_loss: 0.07081099599599838
16:10:12 --- INFO: epoch: 895, batch: 1, loss: 0.7187491655349731, modularity_loss: -0.0936206504702568, purity_loss: 0.7415584921836853, regularization_loss: 0.07081132382154465
16:10:12 --- INFO: epoch: 896, batch: 1, loss: 0.7186645865440369, modularity_loss: -0.09362243860960007, purity_loss: 0.7414751052856445, regularization_loss: 0.07081190496683121
16:10:12 --- INFO: epoch: 897, batch: 1, loss: 0.7185797095298767, modularity_loss: -0.09362504631280899, purity_loss: 0.7413926720619202, regularization_loss: 0.07081207633018494
16:10:12 --- INFO: epoch: 898, batch: 1, loss: 0.7184950709342957, modularity_loss: -0.09362835437059402, purity_loss: 0.7413116097450256, regularization_loss: 0.07081182301044464
16:10:12 --- INFO: epoch: 899, batch: 1, loss: 0.7184106111526489, modularity_loss: -0.09363206475973129, purity_loss: 0.741231381893158, regularization_loss: 0.07081131637096405
16:10:12 --- INFO: epoch: 900, batch: 1, loss: 0.7183259129524231, modularity_loss: -0.09363416582345963, purity_loss: 0.7411488890647888, regularization_loss: 0.0708112046122551
16:10:12 --- INFO: epoch: 901, batch: 1, loss: 0.7182418704032898, modularity_loss: -0.09363509714603424, purity_loss: 0.7410654425621033, regularization_loss: 0.07081154733896255
16:10:12 --- INFO: epoch: 902, batch: 1, loss: 0.7181576490402222, modularity_loss: -0.09363725036382675, purity_loss: 0.7409830689430237, regularization_loss: 0.07081183046102524
16:10:12 --- INFO: epoch: 903, batch: 1, loss: 0.7180736064910889, modularity_loss: -0.0936407595872879, purity_loss: 0.7409024238586426, regularization_loss: 0.0708119124174118
16:10:12 --- INFO: epoch: 904, batch: 1, loss: 0.717989444732666, modularity_loss: -0.09364516288042068, purity_loss: 0.7408229112625122, regularization_loss: 0.07081171870231628
16:10:12 --- INFO: epoch: 905, batch: 1, loss: 0.7179052829742432, modularity_loss: -0.09364978969097137, purity_loss: 0.7407437562942505, regularization_loss: 0.07081131637096405
16:10:12 --- INFO: epoch: 906, batch: 1, loss: 0.7178221344947815, modularity_loss: -0.09365390986204147, purity_loss: 0.7406651377677917, regularization_loss: 0.07081089913845062
16:10:12 --- INFO: epoch: 907, batch: 1, loss: 0.7177382111549377, modularity_loss: -0.09365547448396683, purity_loss: 0.7405826449394226, regularization_loss: 0.07081104069948196
16:10:12 --- INFO: epoch: 908, batch: 1, loss: 0.7176543474197388, modularity_loss: -0.09365524351596832, purity_loss: 0.7404978275299072, regularization_loss: 0.07081173360347748
16:10:12 --- INFO: epoch: 909, batch: 1, loss: 0.7175713777542114, modularity_loss: -0.09365610033273697, purity_loss: 0.7404151558876038, regularization_loss: 0.07081234455108643
16:10:12 --- INFO: epoch: 910, batch: 1, loss: 0.7174882292747498, modularity_loss: -0.09365858137607574, purity_loss: 0.740334153175354, regularization_loss: 0.0708126425743103
16:10:12 --- INFO: epoch: 911, batch: 1, loss: 0.7174049019813538, modularity_loss: -0.09366261959075928, purity_loss: 0.7402549982070923, regularization_loss: 0.07081253081560135
16:10:12 --- INFO: epoch: 912, batch: 1, loss: 0.7173217535018921, modularity_loss: -0.09366749227046967, purity_loss: 0.7401771545410156, regularization_loss: 0.07081209123134613
16:10:12 --- INFO: epoch: 913, batch: 1, loss: 0.7172391414642334, modularity_loss: -0.09367232024669647, purity_loss: 0.7400999069213867, regularization_loss: 0.07081153243780136
16:10:12 --- INFO: epoch: 914, batch: 1, loss: 0.7171564698219299, modularity_loss: -0.09367474913597107, purity_loss: 0.7400197386741638, regularization_loss: 0.07081148028373718
16:10:12 --- INFO: epoch: 915, batch: 1, loss: 0.7170739769935608, modularity_loss: -0.09367528557777405, purity_loss: 0.7399373054504395, regularization_loss: 0.07081196457147598
16:10:12 --- INFO: epoch: 916, batch: 1, loss: 0.7169919610023499, modularity_loss: -0.09367676079273224, purity_loss: 0.7398563027381897, regularization_loss: 0.07081242650747299
16:10:12 --- INFO: epoch: 917, batch: 1, loss: 0.7169098854064941, modularity_loss: -0.09367963671684265, purity_loss: 0.7397768497467041, regularization_loss: 0.07081267237663269
16:10:12 --- INFO: epoch: 918, batch: 1, loss: 0.7168277502059937, modularity_loss: -0.09368374198675156, purity_loss: 0.7396988868713379, regularization_loss: 0.07081261277198792
16:10:12 --- INFO: epoch: 919, batch: 1, loss: 0.7167457342147827, modularity_loss: -0.09368846565485, purity_loss: 0.7396219372749329, regularization_loss: 0.07081224024295807
16:10:12 --- INFO: epoch: 920, batch: 1, loss: 0.7166640758514404, modularity_loss: -0.09369295090436935, purity_loss: 0.7395452857017517, regularization_loss: 0.07081173360347748
16:10:12 --- INFO: epoch: 921, batch: 1, loss: 0.7165830731391907, modularity_loss: -0.09369669109582901, purity_loss: 0.7394684553146362, regularization_loss: 0.07081129401922226
16:10:12 --- INFO: epoch: 922, batch: 1, loss: 0.7165012955665588, modularity_loss: -0.09369776397943497, purity_loss: 0.7393875122070312, regularization_loss: 0.07081153243780136
16:10:12 --- INFO: epoch: 923, batch: 1, loss: 0.7164204120635986, modularity_loss: -0.09369728714227676, purity_loss: 0.739305317401886, regularization_loss: 0.07081237435340881
16:10:12 --- INFO: epoch: 924, batch: 1, loss: 0.7163398265838623, modularity_loss: -0.09369844943284988, purity_loss: 0.7392250895500183, regularization_loss: 0.07081315666437149
16:10:12 --- INFO: epoch: 925, batch: 1, loss: 0.7162590026855469, modularity_loss: -0.09370174258947372, purity_loss: 0.7391471862792969, regularization_loss: 0.07081358879804611
16:10:12 --- INFO: epoch: 926, batch: 1, loss: 0.7161781191825867, modularity_loss: -0.09370672702789307, purity_loss: 0.7390713691711426, regularization_loss: 0.07081349194049835
16:10:12 --- INFO: epoch: 927, batch: 1, loss: 0.7160975337028503, modularity_loss: -0.0937124639749527, purity_loss: 0.7389971017837524, regularization_loss: 0.07081291079521179
16:10:12 --- INFO: epoch: 928, batch: 1, loss: 0.7160175442695618, modularity_loss: -0.0937177911400795, purity_loss: 0.7389232516288757, regularization_loss: 0.07081207633018494
16:10:12 --- INFO: epoch: 929, batch: 1, loss: 0.7159380912780762, modularity_loss: -0.09372199326753616, purity_loss: 0.7388487458229065, regularization_loss: 0.07081136107444763
16:10:12 --- INFO: epoch: 930, batch: 1, loss: 0.715857744216919, modularity_loss: -0.0937231183052063, purity_loss: 0.7387693524360657, regularization_loss: 0.07081150263547897
16:10:12 --- INFO: epoch: 931, batch: 1, loss: 0.7157782316207886, modularity_loss: -0.09372246265411377, purity_loss: 0.7386881709098816, regularization_loss: 0.07081250846385956
16:10:12 --- INFO: epoch: 932, batch: 1, loss: 0.7156994342803955, modularity_loss: -0.09372337907552719, purity_loss: 0.7386091947555542, regularization_loss: 0.0708136186003685
16:10:12 --- INFO: epoch: 933, batch: 1, loss: 0.7156202793121338, modularity_loss: -0.09372642636299133, purity_loss: 0.7385323643684387, regularization_loss: 0.0708143413066864
16:10:12 --- INFO: epoch: 934, batch: 1, loss: 0.7155410647392273, modularity_loss: -0.09373198449611664, purity_loss: 0.7384589314460754, regularization_loss: 0.07081412523984909
16:10:12 --- INFO: epoch: 935, batch: 1, loss: 0.7154622673988342, modularity_loss: -0.09373795986175537, purity_loss: 0.7383869886398315, regularization_loss: 0.07081324607133865
16:10:12 --- INFO: epoch: 936, batch: 1, loss: 0.7153840065002441, modularity_loss: -0.09374317526817322, purity_loss: 0.7383151054382324, regularization_loss: 0.07081209123134613
16:10:12 --- INFO: epoch: 937, batch: 1, loss: 0.7153058052062988, modularity_loss: -0.09374527633190155, purity_loss: 0.7382394671440125, regularization_loss: 0.07081162929534912
16:10:12 --- INFO: epoch: 938, batch: 1, loss: 0.7152278423309326, modularity_loss: -0.09374690055847168, purity_loss: 0.7381630539894104, regularization_loss: 0.07081171870231628
16:10:12 --- INFO: epoch: 939, batch: 1, loss: 0.7151505351066589, modularity_loss: -0.09374890476465225, purity_loss: 0.7380870580673218, regularization_loss: 0.07081236690282822
16:10:12 --- INFO: epoch: 940, batch: 1, loss: 0.7150729298591614, modularity_loss: -0.09375251084566116, purity_loss: 0.7380123734474182, regularization_loss: 0.07081308960914612
16:10:12 --- INFO: epoch: 941, batch: 1, loss: 0.7149962186813354, modularity_loss: -0.0937572494149208, purity_loss: 0.7379399538040161, regularization_loss: 0.07081348448991776
16:10:12 --- INFO: epoch: 942, batch: 1, loss: 0.7149192094802856, modularity_loss: -0.09375973045825958, purity_loss: 0.7378649711608887, regularization_loss: 0.07081396132707596
16:10:12 --- INFO: epoch: 943, batch: 1, loss: 0.7148427963256836, modularity_loss: -0.09376025944948196, purity_loss: 0.737788736820221, regularization_loss: 0.0708143413066864
16:10:12 --- INFO: epoch: 944, batch: 1, loss: 0.7147667407989502, modularity_loss: -0.0937623679637909, purity_loss: 0.7377151250839233, regularization_loss: 0.07081395387649536
16:10:12 --- INFO: epoch: 945, batch: 1, loss: 0.7146903276443481, modularity_loss: -0.09376627951860428, purity_loss: 0.7376435399055481, regularization_loss: 0.07081309705972672
16:10:12 --- INFO: epoch: 946, batch: 1, loss: 0.7146149277687073, modularity_loss: -0.09377171099185944, purity_loss: 0.7375744581222534, regularization_loss: 0.07081218808889389
16:10:12 --- INFO: epoch: 947, batch: 1, loss: 0.714539647102356, modularity_loss: -0.09377596527338028, purity_loss: 0.7375035881996155, regularization_loss: 0.07081202417612076
16:10:12 --- INFO: epoch: 948, batch: 1, loss: 0.7144640684127808, modularity_loss: -0.09377817809581757, purity_loss: 0.7374293804168701, regularization_loss: 0.07081287354230881
16:10:12 --- INFO: epoch: 949, batch: 1, loss: 0.7143888473510742, modularity_loss: -0.0937798023223877, purity_loss: 0.7373544573783875, regularization_loss: 0.07081416249275208
16:10:12 --- INFO: epoch: 950, batch: 1, loss: 0.7143136858940125, modularity_loss: -0.09378296881914139, purity_loss: 0.7372819185256958, regularization_loss: 0.07081472873687744
16:10:12 --- INFO: epoch: 951, batch: 1, loss: 0.7142380475997925, modularity_loss: -0.0937875434756279, purity_loss: 0.7372112274169922, regularization_loss: 0.0708143413066864
16:10:12 --- INFO: epoch: 952, batch: 1, loss: 0.7141631245613098, modularity_loss: -0.09379294514656067, purity_loss: 0.73714280128479, regularization_loss: 0.07081326842308044
16:10:12 --- INFO: epoch: 953, batch: 1, loss: 0.7140882611274719, modularity_loss: -0.09379661828279495, purity_loss: 0.7370723485946655, regularization_loss: 0.07081253081560135
16:10:12 --- INFO: epoch: 954, batch: 1, loss: 0.7140129804611206, modularity_loss: -0.09379874169826508, purity_loss: 0.7369990944862366, regularization_loss: 0.0708126425743103
16:10:12 --- INFO: epoch: 955, batch: 1, loss: 0.7139390707015991, modularity_loss: -0.09380023926496506, purity_loss: 0.7369256019592285, regularization_loss: 0.07081369310617447
16:10:12 --- INFO: epoch: 956, batch: 1, loss: 0.7138649821281433, modularity_loss: -0.0938037857413292, purity_loss: 0.7368538975715637, regularization_loss: 0.07081489264965057
16:10:12 --- INFO: epoch: 957, batch: 1, loss: 0.713790774345398, modularity_loss: -0.09380926191806793, purity_loss: 0.7367843985557556, regularization_loss: 0.07081566005945206
16:10:12 --- INFO: epoch: 958, batch: 1, loss: 0.713716983795166, modularity_loss: -0.09381574392318726, purity_loss: 0.7367171049118042, regularization_loss: 0.07081560045480728
16:10:12 --- INFO: epoch: 959, batch: 1, loss: 0.7136440277099609, modularity_loss: -0.09382234513759613, purity_loss: 0.7366513609886169, regularization_loss: 0.07081502676010132
16:10:12 --- INFO: epoch: 960, batch: 1, loss: 0.7135726809501648, modularity_loss: -0.093827985227108, purity_loss: 0.7365864515304565, regularization_loss: 0.07081422954797745
16:10:12 --- INFO: epoch: 961, batch: 1, loss: 0.7135002017021179, modularity_loss: -0.09383013099431992, purity_loss: 0.7365163564682007, regularization_loss: 0.07081398367881775
16:10:12 --- INFO: epoch: 962, batch: 1, loss: 0.7134277820587158, modularity_loss: -0.09382978826761246, purity_loss: 0.7364432215690613, regularization_loss: 0.07081437110900879
16:10:12 --- INFO: epoch: 963, batch: 1, loss: 0.7133567333221436, modularity_loss: -0.0938306599855423, purity_loss: 0.7363725304603577, regularization_loss: 0.0708148330450058
16:10:12 --- INFO: epoch: 964, batch: 1, loss: 0.7132857441902161, modularity_loss: -0.09383360296487808, purity_loss: 0.7363042235374451, regularization_loss: 0.07081512361764908
16:10:12 --- INFO: epoch: 965, batch: 1, loss: 0.713214635848999, modularity_loss: -0.09383849054574966, purity_loss: 0.7362380623817444, regularization_loss: 0.0708150640130043
16:10:12 --- INFO: epoch: 966, batch: 1, loss: 0.7131443023681641, modularity_loss: -0.09384451806545258, purity_loss: 0.7361741662025452, regularization_loss: 0.07081463932991028
16:10:12 --- INFO: epoch: 967, batch: 1, loss: 0.7130745649337769, modularity_loss: -0.0938485786318779, purity_loss: 0.7361087203025818, regularization_loss: 0.07081441581249237
16:10:12 --- INFO: epoch: 968, batch: 1, loss: 0.7130044102668762, modularity_loss: -0.09385059028863907, purity_loss: 0.7360405325889587, regularization_loss: 0.07081447541713715
16:10:12 --- INFO: epoch: 969, batch: 1, loss: 0.7129347920417786, modularity_loss: -0.09385137259960175, purity_loss: 0.7359713315963745, regularization_loss: 0.07081484794616699
16:10:12 --- INFO: epoch: 970, batch: 1, loss: 0.7128657102584839, modularity_loss: -0.09385407716035843, purity_loss: 0.7359047532081604, regularization_loss: 0.07081501185894012
16:10:12 --- INFO: epoch: 971, batch: 1, loss: 0.7127963900566101, modularity_loss: -0.09385871142148972, purity_loss: 0.7358400821685791, regularization_loss: 0.07081502676010132
16:10:12 --- INFO: epoch: 972, batch: 1, loss: 0.7127273678779602, modularity_loss: -0.09386443346738815, purity_loss: 0.735776960849762, regularization_loss: 0.0708148330450058
16:10:12 --- INFO: epoch: 973, batch: 1, loss: 0.7126587629318237, modularity_loss: -0.09386827051639557, purity_loss: 0.7357120513916016, regularization_loss: 0.07081501185894012
16:10:12 --- INFO: epoch: 974, batch: 1, loss: 0.7125908136367798, modularity_loss: -0.09387015551328659, purity_loss: 0.7356454133987427, regularization_loss: 0.0708155483007431
16:10:12 --- INFO: epoch: 975, batch: 1, loss: 0.7125230431556702, modularity_loss: -0.09387269616127014, purity_loss: 0.7355798482894897, regularization_loss: 0.07081589102745056
16:10:12 --- INFO: epoch: 976, batch: 1, loss: 0.7124553918838501, modularity_loss: -0.09387621283531189, purity_loss: 0.7355155944824219, regularization_loss: 0.07081599533557892
16:10:12 --- INFO: epoch: 977, batch: 1, loss: 0.7123879194259644, modularity_loss: -0.09388050436973572, purity_loss: 0.7354524731636047, regularization_loss: 0.07081592828035355
16:10:12 --- INFO: epoch: 978, batch: 1, loss: 0.7123208045959473, modularity_loss: -0.09388517588376999, purity_loss: 0.7353901863098145, regularization_loss: 0.07081577926874161
16:10:12 --- INFO: epoch: 979, batch: 1, loss: 0.7122541666030884, modularity_loss: -0.09388962388038635, purity_loss: 0.7353280782699585, regularization_loss: 0.07081570476293564
16:10:12 --- INFO: epoch: 980, batch: 1, loss: 0.7121874094009399, modularity_loss: -0.09389147907495499, purity_loss: 0.7352627515792847, regularization_loss: 0.07081616669893265
16:10:12 --- INFO: epoch: 981, batch: 1, loss: 0.7121212482452393, modularity_loss: -0.09389349073171616, purity_loss: 0.7351981997489929, regularization_loss: 0.07081656157970428
16:10:12 --- INFO: epoch: 982, batch: 1, loss: 0.7120553255081177, modularity_loss: -0.09389609843492508, purity_loss: 0.7351347208023071, regularization_loss: 0.07081671059131622
16:10:12 --- INFO: epoch: 983, batch: 1, loss: 0.7119896411895752, modularity_loss: -0.09389951080083847, purity_loss: 0.7350725531578064, regularization_loss: 0.07081657648086548
16:10:12 --- INFO: epoch: 984, batch: 1, loss: 0.7119245529174805, modularity_loss: -0.0939035639166832, purity_loss: 0.735011875629425, regularization_loss: 0.07081623375415802
16:10:12 --- INFO: epoch: 985, batch: 1, loss: 0.7118593454360962, modularity_loss: -0.09390591830015182, purity_loss: 0.7349489331245422, regularization_loss: 0.07081631571054459
16:10:12 --- INFO: epoch: 986, batch: 1, loss: 0.711794912815094, modularity_loss: -0.09390715509653091, purity_loss: 0.7348852753639221, regularization_loss: 0.07081679999828339
16:10:12 --- INFO: epoch: 987, batch: 1, loss: 0.7117305397987366, modularity_loss: -0.09391013532876968, purity_loss: 0.7348236441612244, regularization_loss: 0.07081705331802368
16:10:12 --- INFO: epoch: 988, batch: 1, loss: 0.7116662263870239, modularity_loss: -0.09391487389802933, purity_loss: 0.7347640991210938, regularization_loss: 0.07081697136163712
16:10:12 --- INFO: epoch: 989, batch: 1, loss: 0.7116020917892456, modularity_loss: -0.09392067044973373, purity_loss: 0.7347061634063721, regularization_loss: 0.07081656903028488
16:10:12 --- INFO: epoch: 990, batch: 1, loss: 0.7115387916564941, modularity_loss: -0.09392647445201874, purity_loss: 0.7346492409706116, regularization_loss: 0.0708160400390625
16:10:13 --- INFO: epoch: 991, batch: 1, loss: 0.711475133895874, modularity_loss: -0.09392940253019333, purity_loss: 0.7345883846282959, regularization_loss: 0.07081612199544907
16:10:13 --- INFO: epoch: 992, batch: 1, loss: 0.7114124298095703, modularity_loss: -0.0939299687743187, purity_loss: 0.7345255017280579, regularization_loss: 0.07081688940525055
16:10:13 --- INFO: epoch: 993, batch: 1, loss: 0.7113502621650696, modularity_loss: -0.09393168240785599, purity_loss: 0.734464168548584, regularization_loss: 0.07081775367259979
16:10:13 --- INFO: epoch: 994, batch: 1, loss: 0.7112884521484375, modularity_loss: -0.09393499046564102, purity_loss: 0.7344050407409668, regularization_loss: 0.07081837952136993
16:10:13 --- INFO: epoch: 995, batch: 1, loss: 0.7112263441085815, modularity_loss: -0.09393953531980515, purity_loss: 0.734347403049469, regularization_loss: 0.0708184614777565
16:10:13 --- INFO: epoch: 996, batch: 1, loss: 0.7111647129058838, modularity_loss: -0.09394443780183792, purity_loss: 0.7342911958694458, regularization_loss: 0.07081794738769531
16:10:13 --- INFO: epoch: 997, batch: 1, loss: 0.7111031413078308, modularity_loss: -0.09394896030426025, purity_loss: 0.7342349290847778, regularization_loss: 0.07081715017557144
16:10:13 --- INFO: epoch: 998, batch: 1, loss: 0.7110422253608704, modularity_loss: -0.09395254403352737, purity_loss: 0.7341782450675964, regularization_loss: 0.0708165392279625
16:10:13 --- INFO: epoch: 999, batch: 1, loss: 0.7109813690185547, modularity_loss: -0.09395528584718704, purity_loss: 0.73412024974823, regularization_loss: 0.07081642746925354
16:10:13 --- INFO: epoch: 1000, batch: 1, loss: 0.7109206914901733, modularity_loss: -0.09395747631788254, purity_loss: 0.7340613007545471, regularization_loss: 0.07081687450408936
16:10:13 --- INFO: Training process end.
16:10:13 --- INFO: Evaluating process start.
16:10:13 --- INFO: Evaluation loss, {'modularity_loss': -0.09395743161439896, 'purity_loss': 0.7339998483657837, 'regularization_loss': 0.07081811130046844, 'total_loss': 0.710860550403595}
16:10:13 --- INFO: Evaluating process end.
16:10:13 --- INFO: Predicting process start.
16:10:13 --- INFO: Generating prediction results for Simulation.
16:10:13 --- INFO: Predicting process end.
16:10:13 --- INFO: --------------------- GNN end ---------------------- 
16:10:13 --- INFO: ----------------- Niche trajectory ------------------ 
16:10:13 --- INFO: Calculating NTScore for each niche.
16:10:13 --- INFO: Finding niche trajectory with maximum connectivity using Brute Force.
16:10:13 --- INFO: Calculating NTScore for each niche cluster based on the trajectory path.
16:10:13 --- INFO: Projecting NTScore from niche-level to cell-level.
16:10:13 --- INFO: Output NTScore tables.
16:10:13 --- INFO: --------------- Niche trajectory end ---------------- 
```
:::

```{note}
Please refer to the [Parameters Explanation](./parameters.md) for details about the parameters used here.
```

```{note}
Please refer to the [IO Files Explanation](./IO_files.md) for details of input and output files.
```

## Visualization

```{code-cell}
:class: input-cell

ONTraC_analysis \
    -o analysis_output/simulation \
    -l log/simulation.log \
    --NN-dir output/simulation_NN \
    --GNN-dir output/simulation_GNN \
    --NT-dir output/simulation_NT \
    -r
```

```{code-block}
:class: output-cell

##################################################################################

         ▄▄█▀▀██   ▀█▄   ▀█▀ █▀▀██▀▀█                   ▄▄█▀▀▀▄█
        ▄█▀    ██   █▀█   █     ██    ▄▄▄ ▄▄   ▄▄▄▄   ▄█▀     ▀
        ██      ██  █ ▀█▄ █     ██     ██▀ ▀▀ ▀▀ ▄██  ██
        ▀█▄     ██  █   ███     ██     ██     ▄█▀ ██  ▀█▄      ▄
         ▀▀█▄▄▄█▀  ▄█▄   ▀█    ▄██▄   ▄██▄    ▀█▄▄▀█▀  ▀▀█▄▄▄▄▀

                        version: 1.2.0

##################################################################################
19:54:56 --- WARNING: The --meta-input option is no longer required for ONTraC analysis. This option will be deprecated from v2.0.
19:54:56 --- INFO: Creating directory: analysis_output/simulation
19:54:56 --- INFO: ------------------ RUN params memo ------------------ 
19:54:56 --- INFO:             -------- I/O options -------             
19:54:56 --- INFO: Niche network output directory:  output/simulation_NN
19:54:56 --- INFO: GNN output directory:  output/simulation_GNN
19:54:56 --- INFO: Niche trajectory output directory:  output/simulation_NT
19:54:56 --- INFO: Output directory:  analysis_output/simulation
19:54:56 --- INFO: Log file:  log/simulation.log
19:54:56 --- INFO: ---------------- Visualization options ----------------
19:54:56 --- INFO: Reverse the NT score during visualization: True
19:54:56 --- INFO: Plot each sample separately: False
19:54:56 --- INFO: Scale factor control the size of spatial-based plots: 1.0
19:54:56 --- INFO: ---------------- Suppress options ----------------
19:54:56 --- INFO: Suppress the cell type composition visualization: False
19:54:56 --- INFO: Suppress the niche cluster loadings visualization: False
19:54:56 --- INFO: Suppress the niche trajectory related visualization: False
19:54:56 --- INFO: --------------- RUN params memo end ----------------- 
19:54:56 --- INFO: -------------- Analysis pipeline start -------------- 
19:54:56 --- INFO: Analysis pipeline step 0: load data.
19:54:56 --- INFO: Analysis pipeline step 1: train loss visualization.
19:54:58 --- INFO: Analysis pipeline step 2: spatial-based visualization.
19:55:00 --- INFO: Analysis pipeline step 3: niche cluster visualization.
/sc/arion/work/wangw32/conda-env/envs/ONTraC/lib/python3.11/site-packages/ONTraC/analysis/niche_cluster.py:1047: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
19:55:01 --- INFO: Analysis pipeline step 4: cell type-based visualization.
19:55:03 --- INFO: --------------- Analysis pipeline end --------------- 
```

```{note}
Now you can check the visualization outputs under `analysis_output/simulation` directory.
```
