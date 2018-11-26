# Xray image enhancement

## Preparation
First of all, clone the code
```
git clone https://github.com/shuli0808/X-ray_Super-Resolution.git
```
Then, create a folder:
```
cd X-ray_Super-Resolution && mkdir data
```

### prerequisites
* Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

### data Preparation
* Download the dataset and move it into `data/`

## Train
Look at the options in `trainval_net.py`:
```bash
python trainval_net.py
```

## Test
Look at the options in `test_net.py`:
```bash
python test_net.py
```

## Note
Right now need to fix the size for the image, determine input, output size




