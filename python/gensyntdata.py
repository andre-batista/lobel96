import numpy as np
import model as md
import experiment as exp

setup_filename = 'setup_exp01_basic'
setup_filepath = ''

experiment_name = 'exp_test'
experiment_filepath = ''

myexp = exp.Experiment(setup_filename,setup_filepath,experiment_name,experiment_filepath)
myexp.set_map('square',2,.05)
myexp.plot_map('jpeg')