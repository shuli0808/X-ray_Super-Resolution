# Xray image enhancement

## Requirements
I using python3. Install the dependencies as follows
```
pip install -r requirements.txt
```

## Download the xray dataset
Run the script `build_dataset.py` to crop and save the images. By default, these images will locate at `data/preprocess`:
```bash
python build_dataset.py --data_dir data/xray_images --output_dir data/preprocess --num_val 5000
```




