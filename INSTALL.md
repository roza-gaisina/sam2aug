# Setup

This document explains how to set up the `sam2aug` repository and its external dependencies.

The repository does **not** provide the full third-party codebases for SAM2 or LaMa. These must be installed or made available separately.

## Tested environment

The codebase was developed and tested with:

- Python 3.10.19
- PyTorch 2.9.0 (CUDA 13.0, cu130)
- torchvision 0.24.0 (cu130)
- NVIDIA CUDA runtime 13.0

Key CUDA-related packages:

- nvidia-cuda-runtime==13.0.48
- nvidia-cudnn-cu13==9.13.0.50
- nvidia-nccl-cu13==2.27.7

## 1. Create and activate a Python environment

You can use provided `sam2aug_environment.yml` file to create your environment:

```bash
conda env create -f sam2aug_environment.yml
conda activate <environment_name>
```
## 2. Install external dependency: SAM2

Clone the SAM2 repository separately and install it in editable mode from the SAM2 repository root:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```
Check [text](https://github.com/facebookresearch/sam2) for more details on installation and where to download the checkpoints. 

This makes the `sam2` Python package importable for `segmenter.py`.

You can verify the installation with:

```bash
python -c "import sam2; print(sam2.__file__)"
```

## 3. Install external dependency: LaMa

Clone the LaMa repository separately. Check [text](https://github.com/advimman/lama) for more details on installation and where to download the checkpoints. 

The pipeline uses the `saicinpainting` module from LaMa. In the current setup, the simplest way to make it available is to add the LaMa repository root to `PYTHONPATH`.

Set:

```bash
export LAMA_HOME=/path/to/lama
export PYTHONPATH=$LAMA_HOME:$PYTHONPATH
```

You can verify that LaMa is visible to Python with:

```bash
python -c "from saicinpainting.evaluation.utils import move_to_device; print('LaMa import ok')"
```

## 4. Set required environment variables

The pipeline uses environment variables to configure external dependency paths and output locations.

### Required

```bash
export SAM2_HOME=/path/to/sam2_repo
export LAMA_HOME=/path/to/lama
```

## 5. Install the `sam2aug` package

Run this from the repository root:

```bash
git clone https://github.com/roza-gaisina/sam2aug.git 
cd sam2aug
pip install -e .
```

This installs the package in editable mode, so local code changes are reflected immediately.

To verify that the correct package is imported:

```bash
python -c "import sam2aug; print(sam2aug.__file__)"
python -c "from sam2aug import AugmentationPipeline, Segmenter, LamaInpainter; print('imports ok')"
```

### Recommended combined setup

```bash
export SAM2_HOME=/path/to/sam2_repo
export LAMA_HOME=/path/to/lama
export PYTHONPATH=$SAM2_HOME:$LAMA_HOME:$PYTHONPATH
```

## 6. Make the environment variables persistent

If you do not want to set the variables in every new shell session, add them to your shell startup file.

For Bash:

```bash
echo 'export SAM2_HOME=/path/to/sam2_repo' >> ~/.bashrc
echo 'export LAMA_HOME=/path/to/lama' >> ~/.bashrc
echo 'export SAM2AUG_OUTPUT_DIR=/path/to/output_dir' >> ~/.bashrc
echo 'export PYTHONPATH=$SAM2_HOME:$LAMA_HOME:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

For Zsh, use `~/.zshrc` instead.

## 7. Verify the active environment configuration

To check the current environment variables:

```bash
echo $SAM2_HOME
echo $LAMA_HOME
echo $SAM2AUG_OUTPUT_DIR
```

To inspect the current `PYTHONPATH`:

```bash
echo $PYTHONPATH
echo $PYTHONPATH | tr ':' '\n'
```

To inspect the Python import path actually used at runtime:

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

## 8. Smoke test

After setup, run a minimal smoke test to confirm that the package imports and the core pipeline components can be instantiated.

At minimum, verify:

```bash
python -c "import sam2aug; print(sam2aug.__file__)"
python -c "from sam2aug import AugmentationPipeline, Segmenter, LamaInpainter; print('imports ok')"
python -c "from saicinpainting.evaluation.utils import move_to_device; print('LaMa import ok')"
python -c "import sam2; print(sam2.__file__)"
```

Or use the provided test_pipeline.py script provided in the repository.

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'saicinpainting'`

LaMa is not visible to Python.

Fix:

```bash
export LAMA_HOME=/path/to/lama
export PYTHONPATH=$LAMA_HOME:$PYTHONPATH
```

Then test:

```bash
python -c "from saicinpainting.evaluation.utils import move_to_device; print('LaMa import ok')"
```

### `ModuleNotFoundError: No module named 'sam2'`

SAM2 is not installed in the active environment.

Fix:

```bash
cd /path/to/sam2_repo
pip install -e .
```

### `import sam2aug` imports the wrong repository version

A previous editable install may still be active.

Fix:

```bash
pip uninstall sam2aug
pip install -e .
python -c "import sam2aug; print(sam2aug.__file__)"
```

### Environment variables disappear in a new shell

Add them to `~/.bashrc` or `~/.zshrc` as described above.

## 11. Repository-specific note

All experiment-specific parameters, model checkpoint paths, and output locations should be reviewed in `config.py` and the experiment configuration files before running large-scale generation or evaluation jobs.
