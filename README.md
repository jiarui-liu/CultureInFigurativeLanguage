

Rednote OCR: https://huggingface.co/rednote-hilab/dots.ocr


## Set CULTURE_ROOT globally in the conda env

```bash
# 1. Create activation hook
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# 2. Set env var on activate
echo 'export CULTURE_ROOT=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage' \
  > $CONDA_PREFIX/etc/conda/activate.d/culture_root.sh

# (optional) unset on deactivate
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset CULTURE_ROOT' \
  > $CONDA_PREFIX/etc/conda/deactivate.d/culture_root.sh

# 3. Reload env
conda deactivate && conda activate culture

# 4. Check
echo $CULTURE_ROOT
```

install packages

```
pip install -e .
```

# Dataset Construction

## English Idioms

merge_en_idioms.py

