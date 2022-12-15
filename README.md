# Template Matching

This  repository provides scripts for automatic and manual feature matching.

## Environment setup

The dependencies are in `environment.yml`, so you can use
```
conda env create -f environment.yml (for Linux)
```
to install the dependencies in your virtual environment. This step assumes that you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

This will create you a conda environment called `temp`.

Please activate your environment in terminal:
```
conda activate temp
```

You may install the library using the below command in the root directory:
```
pip install -e .
```

## Usage of the scripts

To see how to draw shapes on the image using OpenCV:

```
python -m scripts.opencv_roi_drawing.py --query_image_path data/left1.png --reference_image_path data/reference.jpg --shape circle
```

To see how OpenCV feature matching works:

```
python -m scripts.template_matching.py --query_image_path data/left1.png --reference_image_path data/reference.jpg
```

To do feature matching manually:

```
python -m scripts.match_images.py --query_image_path data/left1.png --reference_image_path data/reference.jpg
```

To transform all the query images onto the reference image:

```
python -m scripts.transform_all_images.py --data_folder data --reference_image data/reference.jpg
```


