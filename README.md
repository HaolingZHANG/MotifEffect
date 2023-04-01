# TODO Title

TODO abstract

## Installation
You may clone this code repository (using the following command line) or 
just download this code repository with the ZIP format.  

```sh
git clone https://github.com/HaolingZHANG/MotifEffect.git
```


It requires a python version at least 3.7, and some well-established libraries listed in 
[requirements file](https://github.com/HaolingZHANG/MotifEffect/blob/main/requirements.txt).

## Experiments
In this work, three experiments are executed.
Among them, we
1. estimate the representational capability and the numerical stability of three motif populations (incoherent loops, coherent loops, and colliders) within a restricted parameter and definition domain.
2. analyze the trade-off between representational capability and numerical stability based on the set theory and a customized method called maximum-minimum loss search.
3. analyze training performance and robustness of neural networks produced by the default neuroevolution method (i.e. NEAT) and its three variants based on different motif generation tendency.

In this code repository, all experimental designs and parameter settings are clear (see 
[experiments.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/experiments.py)
and
[NEAT configure description](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/confs/README.md)).
In just a few months, you can definitely obtain our consistent or similar 
(due to randomization) raw files in 
[raw folder](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/raw/README.md) 
through executing the script.
Meanwhile, all the figures and videos are created by 
[show_main.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/show_main.py) 
and
[show_supp.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/show_supp.py), 
which means that there is no anthropogenic data filtering or adjusting in the presentation.

## Repository Structure
```html
├── effect                                         // Source codes of 3-node motif effect experiments.
│    ├── __init__.py                               // Exhibition of class and method calls.
│    ├── networks.py                               // Motif class based on PyTorch.
│    │    ├── RestrictedWeight                     // Restricted weight module.
│    │    ├── RestrictedBias                       // Restricted bias module.
│    │    ├── NeuralMotif                          // Neural Motif module.
│    ├── operations.py                             // Progress monitor and basic operations
│    │    ├── Monitor                              // Monitor which outputting the progress based on current state and total state.
│    │    ├── prepare_data                         // Prepare database through the range of variable and sampling points.
│    │    ├── prepare_motifs                       // Prepare motif based on the selected parameters.
│    │    ├── calculate_landscape                  // Calculate the output landscape of the selected motif.
│    │    ├── calculate_gradients                  // Calculate the gradient matrix of the selected motif.
│    │    ├── generate_qualified_motifs            // Generate qualified motif with specific requirements.
│    │    ├── calculate_motif_differences          // Calculate difference (L1 loss) between motif landscapes.
│    │    ├── calculate_population_differences     // Calculate minimum differences (L1 loss) between landscapes of two motif populations.
│    ├── robustness.py                             // Robustness-related operations.
│    │    ├── evaluate_propagation                 // Evaluate the error propagation through the selected motif.
│    │    ├── estimate_lipschitz_by_motif          // Estimate the Lipschitz constant of the selected motif.
│    │    ├── estimate_lipschitz_by_signals        // Estimate the Lipschitz constant of the output signals produced by selected motif.
│    ├── similarity.py                             // Similarity-related operations.
│    │    ├── maximum_minimum_loss_search          // Find the maximum-minimum L1 loss (as the representation capacity bound) between source motif and target motifs.
│    │    ├── minimum_loss_search                  // Train the target motif to achieve the source motif and find the minimum L1 loss between the two motifs.
├── practice                                       // Source codes of neuroevolution experiments.
│    ├── __init__.py                               // Exhibition of class and method calls.
│    ├── agent.py                                  // Trained agent classes.
│    │    ├── DefaultAgent                         // Default trained agent.
│    │    ├── NEATAgent                            // Trained agent for the NEAT method and its variants.
│    │    ├── create_agent_config                  // Create training configure of agent.
│    │    ├── obtain_best                          // Obtain best genome in the specific task.
│    │    ├── train_and_evaluate                   // Train and evaluate agents in a given NEAT task.
│    │    ├── train                                // Train an agent in a given NEAT task.
│    │    ├── evaluate                             // Evaluate an agent in a given NEAT task.
│    ├── evolve.py                                 // Neuroevolution method variation.
│    │    ├── AdjustedReproduction                 // Adjusted NEAT method (prohibiting the appearance of loops based on the given setting).
│    │    ├── AdjustedGenome                       // Adjusted Genome, agent of NEAT method (prohibiting the appearance of loops based on the given setting).
│    │    ├── AdjustedGenomeConfig                 // Adjusted Genome configuration (prohibiting the appearance of loops based on the given setting).
│    │    ├── create_adjacency_matrix              // Create the adjacency matrix from the given genome and its corresponding configuration.
│    ├── motif.py                                  // Motif-related operations.
│    │    ├── acyclic_motifs                       // Dictionary of 3-node acyclic motifs, i.e. incoherent loops, coherent loops, colliders.
│    │    ├── motif_matrix_to_fingerprint          // Change the type from the motif to the fingerprint for saving.
│    │    ├── fingerprint_to_motif_matrix          // Change the type from the fingerprint to the motif for motif calculating.
│    │    ├── motif_matrix_to_connections          // Change the type from the motif matrix to the graph connections for agent training.
│    │    ├── connections_to_motif_matrix          // Change the type from the graph connections to the motif matrix for motif calculating.
│    │    ├── is_same_motif                        // Judge whether the two motifs are the same.
│    │    ├── obtain_motif                         // Obtain a candidate motif of specific nodes in the adjacency matrix.
│    │    ├── compliance_motif_specification       // Test the rationality of the obtained motif.
│    │    ├── collect_motifs                       // Collect all the rational motifs from the adjacency matrix.
│    │    ├── count_motifs_from_adjacency_matrix   // Count the motif frequencies from a given adjacency matrix.
│    │    ├── detect_motifs_from_adjacency_matrix  // Detect the given motifs from an adjacency matrix
│    ├── task.py                                   // Task modules.
│    │    ├── GymTask                              // OpenAI gym task.
│    │    ├── NEATCartPoleTask                     // CartPole task for NEAT method and its variants.
│    │    ├── NormNoiseGenerator                   // Noise generator based on the norm.
├── works                                          // Experiment module of this work.
│    ├── confs                                     // Configuration folder of neuroevolution tasks.
│    ├── data                                      // Painting data folder of all the experiments.
│    ├── raw                                       // Raw data folder of all the experiments.
│    ├── show                                      // Painted figure folder of all the experiments.
│    ├── __init__                                  // Preset parameters in the experiments.
│    ├── experiments.py                            // Run 3 experiments and generate the painting data.
│    ├── show_main.py                              // Paint figures (in the main text) from the generated data.
│    ├── show_supp.py                              // Paint figures and videos (in the supporting materials) from the generated data.
├── LICENSE                                        // License of this library (GPL-3.0 license).
├── README.md                                      // Description document of this library.
├── requirements.txt                               // Basic library requirements of this library.
```

## Citation
If you think this repository helps or being used in your research, please consider refer this work.
Here is a Bibtex entry:

TODO
