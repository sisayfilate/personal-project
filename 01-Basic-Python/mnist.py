# self made moudle

import pandas as pd 
import numpy as np 

# ------------------------------------------------- #
# Training methods                                  #
# ------------------------------------------------- #

def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):
    """
    Sagemaker passes num_cpus, num_gpus and other args we can use to tailor traning to
    the current container environment, but here we just use simple cpu context.
    """
    ctx = mx.cpu()
    
    
def hello_world(x):
    return f'Hello {x}, we are delighted to see you here.'
    