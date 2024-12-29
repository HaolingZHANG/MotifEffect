# Configuration of NEAT algorithm and its variations

## Basic Setting

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
2. "remove_type" is "c" refers to prohibiting coherent loops 
from being generated during the evolution of neural networks.
3. "remove_type" is "a" refers to prohibiting all the loops (including incoherent loops and coherent loops) 
from being generated during the evolution of neural networks.

At the stage of training agents, "enabled" parameter in ConnectionGene 
does not affect the creation of the adjacency matrix. 
Whether a selected motif is created in the neural network is a fact that will have an impact on subsequent evolution.
The "enabled" parameter may change from "False" to "True", making the prohibition of selected motifs unsuccessful.

On the contrary, for the agent created by best genome after training, 
"enabled" parameter in ConnectionGene is considered in the creation of the adjacency matrix. 
Because the connection to which False "enabled" belongs is not actually used.

## A Reinforcement Learning Case (main folder)

Since we trained the CartPole-v0 environment under varying noise conditions using NEAT and its variants, 
apart from the aforementioned adjustments, 
its configuration file adheres to the settings of the 
[original setting](https://github.com/CodeReclaimers/neat-python/tree/master/examples/single-pole-balancing).

## Other Case studies (supp folder)

We selected three representative real-world datasets to validate the scalability of our conclusions.
These datasets, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/), 
correspond to the domains of 
(1) [biology](https://archive.ics.uci.edu/dataset/39/ecoli), 
(2) [physics and chemistry](https://archive.ics.uci.edu/dataset/42/glass+identification), 
and (3) [medicine and health](https://archive.ics.uci.edu/dataset/212/vertebral+column), 
respectively.

Since these tasks belong to supervised learning, 
the setting of population size follows the setting of 
[NEAT XOR task](https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward).
