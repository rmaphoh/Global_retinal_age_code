## Gloabl Retinal Age

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.


### Data requirements

- To develop local retinal age model, please include ~100k-200k images. The docker running will take around 3-5 days. 

- To support only cross validation, please include ~10k-50k images. The docker running will take around 1-2 days. 


### Data preparation

1. Create and enter a folder for retinal age project, e.g. `Retina_age`.

2. Put the images into `Images` folder (both png and jpg formats are fine).

```
├──Retina_age
    ├──Images
        ├──1.jpg
        ├──2.jpg
        ├──3.jpg
``` 

2. Generate a `metadata.csv` file including the metadata. An example can be found [here](https://drive.google.com/file/d/1tDwguNTdByc7N0CNOmtU6TppRe548P1D/view?usp=sharing).

- The columns of "Patient_id", "Image", and "Age" should be available.

- If some other variables are missing / not known, then leave them BLANK.

- Save the metadata.csv in the same path as `Images` folder

```
├──Retina_age
    ├──Images
        ├──1.jpg
        ├──2.jpg
        ├──3.jpg
    ├──metadata.csv   
``` 


### 🔧Install code environment

1. Create environment with conda:

```
conda create -n retfound_age python=3.11.0 -y
conda activate retfound_age
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/rmaphoh/Global_retinal_age_code.git
cd Global_retinal_age_code
pip install -r requirements.txt
```


### 🌱Training a local model and internal evaluation

1. Download the RETFound-DINOv2 [weight](https://drive.google.com/file/d/1Wd5OuU3jXQbGmojPlGIWqT-p0DhiEQJE/view?usp=sharing) and put it in the project folder `Global_retinal_age_code`.

2. Please substitute the `{Absolute_path}` with the path of `Retina_age` folder created above. If the images have already been pre-processed, comment the `python EyeQ_process_main.py`. 

```
sh train.sh
```

Please zip `metadata.csv` and `output_dir` folder, and share them through storage platforms, e.g. Google Drive and OneDrive.
<br><br><br>


### Hardware requirements

- A consumer-grade GPU (~16GB) is essential for model training. 
