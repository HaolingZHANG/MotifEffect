# Replacing incoherent loops with colliders to improve the robustness of artificial neural networks

## Installation
You can install this package using pip:
```sh
pip install MotifEffect
```
The packages requires a python version >=3.7, 
as well as some basic libraries listed in [requirements file](https://github.com/HaolingZHANG/MotifEffect/blob/main/requirements.txt).

## Repository Structure
```html
├── data                                     // Configures and results in this work.
│    ├── configs                             // Configures for executing NEAT-python framework (and its variants).
│    ├── figures                             // Output figures (in the paper).
│    ├── results                             // Raw results and results for presentation in each task.
├── effect                                   // Source code of 3-node motif experiments.
│    ├── __init__.py                         // Exhibition of class and method calls.
│    ├── networks.py                         // Motif class based on PyTorch.
│    │    ├── RestrictedWeight               // Restricted weight module.
│    │    ├── RestrictedBias                 // Restricted bias module.
│    │    ├── NeuralMotif                    // Neural Motif module.
│    ├── operations.py                       // Progress monitor and basic operations
│    │    ├── Monitor                        // Monitor which outputting the progress based on current state and total state.
│    │    ├── prepare_data                   // Prepare database through the range of variable and sampling points.
│    │    ├── prepare_motifs                 // Prepare motif based on the selected parameters.
│    │    ├── calculate_landscape            // Calculate the output landscape of the selected motif.
│    │    ├── calculate_gradients            // Calculate the gradient matrix of the selected motif.
│    ├── robustness.py                       // Robustness-related operations.
│    │    ├── evaluate_propagation           // Evaluate the error propagation through the selected motif.
│    │    ├── estimate_lipschitz             // Estimate the Lipschitz constant of the selected motif.
│    │    ├── calculate_rugosity             // Calculate rugosity index of the output landscape through the selected motif.
│    ├── similarity.py                       // Similarity-related operations.
│    │    ├── maximum_minimum_loss_search    // Find the maximum-minimum L1 loss (as the representation capacity bound) between source motif and target motifs.
│    │    ├── minimum_loss_search            // Train the target motif to achieve the source motif and find the minimum L1 loss between the two motifs.
├── practice                                 // Source code of neuroevolution experiments.
│    ├── __init__.py                         // Exhibition of class and method calls.
│    ├── agent.py                            // Trained agent classes.
│    │    ├── DefaultAgent                   // Default trained agent.
│    │    ├── NEATAgent                      // Trained agent for the NEAT method and its variants.
│    ├── evolve.py                           // Neuroevolution methods.
│    │    ├── BiReproduction                 // Bi-NEAT method (see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── GSReproduction                 // GS-NEAT method (see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── NSReproduction                 // NEAT method with novelty search.
│    │    ├── AdjustedReproduction           // Adjusted NEAT method (prohibiting the appearance of incoherent loops).
│    │    ├── GlobalGenome                   // Genome for global search (i.e. binary search, see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── GlobalGenomeConfig             // Genome configure for global search (i.e. binary search, see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── UpdatedSpeciesSet              // Species set for global search (i.e. binary search, see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── create_center_new              // Create a new genome at the midpoint of two genomes (see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── create_golden_section_new      // Create a new genome at the golden-section of two genomes, near genome 1. (see https://ieeexplore.ieee.org/abstract/document/9185648/).
│    │    ├── create_near_new                // Create a new genome near the old genome. 
│    │    ├── distance_between_two_matrices  // Obtain the distance between two genome matrices.
│    ├── handle.py                           // Task handles.
│    │    ├── Monitor                        // Monitor which outputting the progress based on current state and total state.
│    │    ├── obtain_best                    // Obtain best genome in the specific task.
│    │    ├── create_agent_config            // Create training configure of agent.
│    │    ├── calculate_matrix_from_agent    // Calculate the adjacency matrix (directed graph) of NEAT agent.
│    │    ├── calculate_agent_frequency      // Calculate the motif frequency of NEAT agents.
│    ├── motif.py                            // Motif-related operations.
│    │    ├── GraphType                      // Collected graph type of matrix.
│    │    ├── motif_matrix_to_fingerprint    // Change the type from the motif to the fingerprint for saving.
│    │    ├── fingerprint_to_motif_matrix    // Change the type from the fingerprint to the motif for motif calculating.
│    │    ├── motif_matrix_to_connections    // Change the type from the motif matrix to the graph connections for agent training.
│    │    ├── connections_to_motif_matrix    // Change the type from the graph connections to the motif matrix for motif calculating.
│    │    ├── is_same_motif                  // Judge whether the two motifs are the same.
│    │    ├── find_same_motifs               // Obtain best genome in the specific task.
│    │    ├── obtain_motif                   // Obtain a candidate motif of specific nodes in the adjacency matrix.
│    │    ├── compliance_motif_specification // Test the rationality of the obtained motif.
│    │    ├── collect_motifs                 // Collect all the rational motifs from the adjacency matrix.
│    │    ├── count_motifs_from_matrices     // Statistics the rational motif frequencies from matrix groups.
│    │    ├── generate_random_graphs         // Generate adjacency matrix of random graphs.
│    │    ├── calculate_z_scores             // Calculate z-scores from the adjacency matrix.
│    ├── robust.py                           // Noise generator and intervention process.
│    │    ├── NormNoiseGenerator             // Noise generator based on the norm.
│    │    ├── intervene                      // Intervene the gym environment in the test process.
│    ├── task.py                             // Task modules.
│    │    ├── GymTask                        // OpenAI gym task.
│    │    ├── NEATCartPoleTask               // CartPole task for NEAT method and its variants.
├── works                                    // Experiment module of this work.
│    ├── __init__                            // Preset parameters in the experiments.
│    ├── experiments.py                      // Five experiments.
│    ├── show_main.py                        // Script showing data in the main text.
│    ├── show_supp.py                        // Script showing data in the supplementary.
├── LICENSE                                  // License of this library (GPL-3.0 license).
├── README.md                                // Description document of this library.
├── requirements.txt                         // basic library requirements of this library.
```

## Experiments
TODO

## Citation
If you think this repository helps or being used in your research, please consider refer this work.
Here is a Bibtex entry:

TODO
