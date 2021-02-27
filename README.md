## Bachelor Thesis
**"Derivative-free Optimization for Sequence-to-Sequence Models"**

Institute of Computational Linguistics  
Heidelberg University, Germany

### Author
Lyuba Dimitrova
lyuba.dimitrova@abv.bg


### Overview
This repository contains my implementation of derivative-free optimization for seq2seq models. It builds on the [JoeyNMT](https://github.com/joeynmt/joeynmt) framework.
Code was tested on Python 3.5 and 3.7.

### Usage

Clone the repository.
Install the requirements with `pip install -r requirements.txt`, preferably in a separate virtualenv or conda environment.

Before running the DF training, build a configuration file. configs/example.yaml provides a description of the available options. Alternatively, you can use configs/reverse.adam.005.yaml for testing purposes. 

Run with:
```
python[3] dfo.py configs/reverse.adam.005.yaml [--debug]
```
Help: 
```
$ python dfo.py --help

usage: dfo.py [-h] [--debug] config_file

positional arguments:
  config_file  The YAML configuration file.

optional arguments:
  -h, --help   show this help message and exit
  --debug      If set, parameters, gradients and update steps are also saved.
```


### Project structure

	|_ dfo.py
	|    Contains the DFTrainManager class, which configures and runs the training process.
    |
    |_ scripts/   
    |    |_ grad_estimators.py - implementation of the GradEstimator class, which provides implementations
    |    |                       for the vanilla, forward FD, and antithetic gradient estimators
    |    |_ helpers.py  - helper functions and dictionaries
    |    |_ optimizers.py  - implementations of SGD, Momentum-SGD and Adam that work with outside gradients.
    |    |_ reward_function.py   - the reward function used in my experiments
    |    |_ __init__.py  
    |
    |_ data/
	|    Contains training, development, test data and backpropagation checkpoints for the 
    |    copy, sort and reverse tasks.
    |
    |_ joeynmt/
	|    Contains a snapshot of the JoeyNMT framework for easier access.
    |
    |_ configs/
    |    Contains the training configuration files.
    |    |_ example.yaml - a (non-usable) configuration file with option explanations
    |    |_ reverse.adam.005.yaml - a (usable) configuration file if you want to run a little test
    |
    |_ DFOseq2seqReport.pdf
    |    My actual thesis.
    |
	|_ README.md
	|    This file.
	|
	|_ LICENSE
	|
    |_ requirements.txt
    |



### Licenses for used software


##### JoeyNMT
https://github.com/joeynmt/joeynmt/blob/master/LICENSE


##### PyTorch 
https://github.com/pytorch/pytorch/blob/master/LICENSE


##### torchtext
https://github.com/pytorch/text/blob/master/LICENSE


#### matplotlib
https://matplotlib.org/3.1.1/users/license.html


#### OpenAI ES starter  (for optimizers.py)
https://github.com/openai/evolution-strategies-starter/blob/master/LICENSE

