# link-prediction
**Follow these steps on the development server (not your local machine).**

# Setup
**Clone this repo with `git clone --recurse-submodules <URI>`.**

## 1. Follow instructions to download llama models
- **NOTE: The repo is already added as a submodule, so no need to clone separately.**
- Instructions per https://github.com/facebookresearch/llama
  - https://ai.meta.com/resources/models-and-libraries/llama-downloads/

## 2. Install miniconda
`mkdir -p ~/miniconda3`

`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh`

`bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`

`rm -rf ~/miniconda3/miniconda.sh`

## 3. Create the conda virtual environment
`conda env create -f environment.yml`

`conda activate link-prediction`

- This will install all dependencies required by this project via `pip` (including the llama repo as an editable package).

# Run
`torchrun --nproc_per_node 2 main.py --path_to_cfg ./experiment_config.json --ckpt_dir /home/parklab/models/llama2/llama-2-13b-chat/ --tokenizer_path /home/parklab/models/llama2/tokenizer.model`

# Miscellaneous
## llama submodule
The Llama repo was added as a submodule via the following. For now, we will use a git submodule to version control the llama editable pip package.
`git submodule add https://github.com/facebookresearch/llama`

`git config -f .gitmodules submodule.llama.ignore dirty # ignore dirty commits in the submodule`

`git config -f .gitmodules submodule.llama.branch main`

If you forgot to clone with `--recurse-submodules`, you'll need to run the following command:

`git submodule update --init --recursive`

### Updating Llama submodule
https://stackoverflow.com/questions/1777854/how-can-i-specify-a-branch-tag-when-adding-a-git-submodule/18799234

```
git submodule foreach -q --recursive \
  'git switch \
  $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo main)'

git submodule update --recursive --remote
```

## Model directory
The model files should be downloaded to a shared directory so the single copy can be used by multiple users. You will point the llama APIs to the model in this directory.

## requirements.txt
The `pytorch` packages were chosen based on https://pytorch.org/get-started/locally/ and selecting:
- Stable
- Linux
- Pip
- Python
- Cuda 12.1
