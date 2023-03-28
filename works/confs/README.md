# Configuration of NEAT algorithm and its variations
The [template](https://github.com/CodeReclaimers/neat-python/blob/master/examples/single-pole-balancing/config-feedforward) 
of the configuration comes from 
[NEAT-Python framework](https://github.com/CodeReclaimers/neat-python/).

For the default NEAT algorithm, some parameters are adjusted:
1. The activation function and its options are adjusted to "tanh", 
to provide better robustness while ensuring learning ability.
2. The value ranges of weight and bias are reduced from [-30, +30] to [-1, +1], 
in order to be consistent with the motif effect experiments.

For the variations, the additional parameter is "remove_type".
1. "remove_type" is "i" refers to prohibiting incoherent loops 
from being generated during the evolution of neural networks.
1. "remove_type" is "c" refers to prohibiting coherent loops 
from being generated during the evolution of neural networks.
1. "remove_type" is "a" refers to prohibiting all the loops (including incoherent loops and coherent loops) 
from being generated during the evolution of neural networks.
