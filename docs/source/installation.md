# Installation and Setup

This tutorial walks you through installing ONTraC (Ordered Niche Trajectory Construction) and configuring your environment for spatial omics data analysis.

## System Requirements

ONTraC supports the following systems:

- Operating Systems: Linux, macOS, and Windows
- Python Versions: 3.10, 3.11, and 3.12
- Optional: CUDA-capable GPU for faster processing (recommended for large datasets)

```{mermaid}
flowchart LR
  subgraph A["System Requirements"]
    B("Operating Systems:

    • Linux
    • macOS
    • Windows")

    C("Python Versions:

    • 3.10
    • 3.11
    • 3.12")

    D("Hardware Recommendations:

    • CUDA-capable GPU (optional but recommended)
    • Sufficient RAM for large datasets")

  end
```

## GPU Configuration

ONTraC can utilize GPU acceleration via CUDA for faster processing. If a CUDA-capable GPU is not available, ONTraC will run on the CPU.

The following PyTorch CUDA versions are supported:

- cu118 (CUDA 11.8)
- cu126 (CUDA 12.6)
- cu128 (CUDA 12.8)

Please refer to the [official CUDA website](https://docs.nvidia.com/cuda/) for CUDA installation instructions.

## Installation

ONTraC can be installed using pip. Choose the installation method that best suits your workflow.

### Step 0: Clear Cache (Optional but Recommended)

```sh
pip cache purge
conda clean -a -y
```

### Step 1: Create and Activate a Conda Environment (Optional but Recommended)

```bash
conda create -y -n ONTraC python=3.11
conda activate ONTraC
```

#### Install Optional Dependencies for PyG (Only works on systems with a CUDA-capable GPU)

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
# or
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
# based on your CUDA version
# or
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
# for macOS and non-CUDA-capable systems
```

### Step 2: Install ONTraC

#### Option 1: Install Stable Version using pip

For basic functionality:

```bash
pip install ONTraC
```

For visualization capabilities:

```bash
pip install "ONTraC[analysis]"
```

For testing capabilities:

```bash
pip install "ONTraC[test]"
```

For development capabilities:

```bash
pip install "ONTraC[dev]"
```

For all capabilities:

```bash
pip install "ONTraC[all]"
```

#### Option 2: Install the Development Version from GitHub

For the latest development version:

```bash
git clone git@github.com:gyuanlab/ONTraC.git .
cd ONTraC
pip install .
# Or with visualization capabilities:
pip install ".[analysis]"
# Or with test capabilities:
pip install ".[test]"
# Or with development capabilities:
pip install ".[dev]"
# Or with all capabilities:
pip install ".[all]"
```

### Step 3: Set Up Jupyter (Optional but Recommended)

If you plan to use ONTraC with Jupyter notebooks, add the ONTraC environment as a new kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name ONTraC --display-name "Python 3.11 (ONTraC)"
```

### Step 4: Test

#### CUDA Availability Test (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```{note}
Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for PyTorch installation instructions.
```

#### ONTraC Installation Test

```{code-block}
:class: input-cell

python -c "import ONTraC; ONTraC.utils.write_version_info()"
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
```
