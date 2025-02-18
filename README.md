# PatchPRO

The code for paper "Patch Synthesis for Property Repair of Deep Neural Networks" appearing in ICSE 2025. 


## Installation

In your virtual environment, either install directly from this repository by
```
git clone git@github.com:XuankangLin/ART.git
cd PatchPRO
pip install -r requirements.txt
```

To replay the evaluation, you should firstly download the data and model from [figshare](https://figshare.com/s/92c8d9bb0ddbc226e6fb), and unzip this file according description.

Then you can run individual python file from `script_file/` . For example, if you want to evaluate the experiment for Acas Xu, you can run  

```
python acasxu_exp.py
```
The corresponding result have been stored in table.csv.
