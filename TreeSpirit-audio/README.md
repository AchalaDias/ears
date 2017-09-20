## Installation

  
### Step 1 - install Python 2.7 using [Berry Conda](https://github.com/jjhelmus/berryconda)

- Install conda for armv7l to `/opt/conda`:

```bash
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
chmod +x Miniconda3-latest-Linux-armv7l.sh
sudo ./Miniconda3-latest-Linux-armv7l.sh
```
 
- Add `export PATH="/opt/conda/bin:$PATH"` to the end of `/home/pi/.bashrc`. Then reload with `source /home/pi/.bashrc`.

- Install Python with required packages:

```bash
conda config -add channels rpi
conda create -n treeSpirite python=2.7
source activate treeSpirite
conda install cython numpy pandas scikit-learn cffi h5py
```

### Step 2 - install requirements

```bash
pip install -r requirements.txt
```


## Training new models

If you want to train the same model on a different dataset:
- Put all audio files (WAV) into `/dataset/audio`.
- Replace the [`/dataset/dataset.csv`](/dataset/dataset.csv) file with new CSV:

```csv
filename,category
```

- Run `python train.py` - this should result in the following files being generated on the server:

File                | Description
------------------- | ------------------------------------------------------- 
`model.h5`          | weights of the learned model
`model.json`        | a serialized architecture of the model (Keras >=2.0.0)  
`model_labels.json` | dataset labels


