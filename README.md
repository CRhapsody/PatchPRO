# PatchPRO

The code for paper "Patch Synthesis for Property Repair of Deep Neural Networks" appearing in ICSE 2025. 


## Installation

In your virtual environment, either install directly from this repository by
```
git clone https://github.com/CRhapsody/PatchPRO.git
cd PatchPRO
pip install -r requirements.txt
```

To replay the evaluation, you should firstly download the data and model from [figshare](https://figshare.com/s/92c8d9bb0ddbc226e6fb), and unzip this file according description.

Then you can run individual python file from `script_file/` . For example, if you want to evaluate the experiment for Acas Xu, you can run  

```
python acasxu_exp.py
```
The corresponding result have been stored in table.csv.

## DOCKER ENVIRONMENT
We provide a docker environment for running the experiments. You can use the following command to build the docker image:
```
docker build -t patchpro:latest .
docker run -it --rm --gpus all -v /path/to/PatchPRO:/home/docker/PatchPRO -v /path/to/data_and_model.zip:/home/docker/PatchPRO/data_and_model.zip patchpro:latest
```

Unzip the data_and_model.zip file in the docker container. Then you can run the experiments as described above.
```
unzip /home/docker/data_and_model.zip -d /home/docker/PatchPRO
```

After extracting the file, you will find two folders: `data` and `model`. Place them inside the `PatchPRO` folder so that their relative paths are `./PatchPRO/data` and `./PatchPRO/model`. Incorrect paths may cause runtime errors. Ensure proper permissions for these directories.