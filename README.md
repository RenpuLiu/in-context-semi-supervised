
## Getting started


1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

```
    conda env create -f environment.yml
```
```
    conda activate in-context-learning
```
3. Run the following code for our main experiments.
```
     python train.py --config conf/semi_supervised.yaml
```
## Our code is based on Garg's work

https://github.com/dtsip/in-context-learning
