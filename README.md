# Representational capacity and numerical stability of network motifs and their role in optimizing artificial neural networks

While network motifs are fully investigated in various networked systems, 
early efforts fell short in recognizing the significance of network motifs to artificial neural networks, 
therefore, motif-based structure optimization remain unexplored territory. 
Hereof, we investigate the properties of incoherent loops, coherent loops, and colliders, 
and extend them to network scale. 
Using colliders as a benchmark, we develop static and dynamic strategies to analyze trade-off tendencies 
between representational capacity and numerical stability of two types of loops, 
finding that incoherent loops outperform coherent loops and 
predicting that replacing coherent loops with colliders can enhance network performance. 
To illustrate their network-scale applicability, we use baseline neuroevolution and 
its finely-modified variants to produce motif-limited networks 
for solving the cart-pole balance task in noisy environments. 
As predicted, we derive consistent findings in network scale. 
These findings suggest a methodology to improve noise resilience by 
regularizing network structure design by biasing towards specific motif palettes.

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
1. analyze trade-off tendencies between representational capacity and numerical stability of two types of loops, using colliders as a benchmark, based on gradient descent strategy and motif-associated Lipschitz constant.
2. analyze the origin of these trade-off tendencies using maximum-minimum loss search.
3. analyze training efficiency and robustness of neural networks produced by the classical neuroevolution method (i.e. NEAT) and its three variants based on different motif generation tendency.

In this code repository, all experimental designs and parameter settings are clear (see 
[run_1_tasks.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/run_1_tasks.py)
and
[NEAT configure description](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/confs/README.md)).
In just a few months, you can definitely obtain our consistent or similar 
(due to randomization) raw files in 
[raw folder](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/raw/README.md) 
through executing the script.
Meanwhile, all the figures and videos are created by 
[show_main.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/show_main.py), 
[show_supp.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/show_supp.py) and
[show_video.py](https://github.com/HaolingZHANG/MotifEffect/blob/main/works/show_video.py) 
which means that there is no anthropogenic data filtering or adjusting in the presentation.

## Repository Structure
```html
├── effect                                         // Source codes of 3-node motif effect experiments.
│    ├── __init__.py                               // Exhibition of class and method calls and implementation of monitor class.
│    ├── networks.py                               // Motif class based on PyTorch.
│    │    ├── RestrictedWeight                     // Restricted weight module.
│    │    ├── RestrictedBias                       // Restricted bias module.
│    │    ├── NeuralMotif                          // Neural Motif module.
│    ├── operations.py                             // Basic operations.
│    │    ├── prepare_data                         // Prepare database through the range of variable and sampling points for both x and y.
│    │    ├── prepare_data_flexible                // Prepare database through the flexible range of variable and sampling points.
│    │    ├── prepare_motifs                       // Prepare motif based on the selected parameters.
│    │    ├── calculate_landscape                  // Calculate the output landscape of the selected motif.
│    │    ├── calculate_values                     // Calculate the output values of the selected motif.
│    │    ├── calculate_gradients                  // Calculate the gradient matrix of the selected motif.
│    │    ├── detect_curvature_feature             // Detect the curvature feature of a given landscape.
│    │    ├── generate_motifs                      // Generate qualified motif with specific requirements.
│    │    ├── generate_outputs                     // Generate all output landscapes and the corresponding parameters based on the given parameter domain.
│    │    ├── calculate_differences                // Calculate norm differences between motif landscapes.
│    ├── robustness.py                             // Robustness-related operations.
│    │    ├── evaluate_propagation                 // Evaluate the error propagation through the selected motif.
│    │    ├── estimate_lipschitz                   // Estimate the Lipschitz constant of the output signals produced by selected motif.
│    │    ├── estimate_lipschitz_by_motif          // Estimate the Lipschitz constant from a selected motif.
│    ├── similarity.py                             // Similarity-related operations.
│    │    ├── execute_escape_processes             // Execute the escape process for multiple pairs of an escape motif and several catch motifs.
│    │    ├── execute_catch_processes              // Execute the catching process for referenced motifs and several catch motifs.
│    │    ├── maximum_minimum_loss_search          // Find the maximum-minimum L2-norm loss (as the representation capacity bound) between source motif and target motifs.
│    │    ├── minimum_loss_search                  // Train the target motif to achieve the source motif and find the minimum L2-norm loss between the two motifs.
├── practice                                       // Source codes of neuroevolution experiments.
│    ├── __init__.py                               // Exhibition of class and method calls.
│    ├── agent.py                                  // Trained agent classes.
│    │    ├── DefaultAgent                         // Default trained agent.
│    │    ├── NEATAgent                            // Trained agent for the NEAT method and its variants.
│    │    ├── create_agent_config                  // Create training configure of agent.
│    │    ├── obtain_best                          // Obtain the best genome in the specific task.
│    │    ├── train_and_evaluate                   // Train and evaluate agents in a given NEAT task.
│    │    ├── train                                // Train an agent in a given NEAT task.
│    │    ├── evaluate                             // Evaluate an agent in a given NEAT task.
│    ├── evolve.py                                 // Neuroevolution method variation.
│    │    ├── AdjustedGenome                       // Adjusted Genome, agent of NEAT method (prohibiting the appearance of loops based on the given setting).
│    │    ├── AdjustedGenomeConfig                 // Adjusted Genome configuration (prohibiting the appearance of loops based on the given setting).
│    │    ├── AdjustedReproduction                 // Adjusted NEAT method (prohibiting the appearance of loops based on the given setting).
│    │    ├── create_adjacency_matrix              // Create the adjacency matrix from the given genome and its corresponding configuration.
│    ├── motif.py                                  // Motif-related operations.
│    │    ├── acyclic_motifs                       // Dictionary of 3-node acyclic motifs, i.e., incoherent loops, coherent loops, colliders.
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
│    ├── noise.py                                  // Noise modules.
│    │    ├── NormNoiseGenerator                   // Noise generator based on the norm.
│    ├── task.py                                   // Task modules.
│    │    ├── GymTask                              // OpenAI gym task.
│    │    ├── NEATCartPoleTask                     // CartPole task for NEAT method and its variants.
├── works                                          // Experiment module of this work.
│    ├── confs                                     // Configuration folder of neuroevolution tasks.
│    ├── data                                      // Painting data folder of all the experiments.
│    ├── raw                                       // Raw data folder of all the experiments.
│    ├── show                                      // Painted figure folder of all the experiments.
│    ├── temp                                      // Temp folder to temporarily save all the figures in Video S1 - S8.
│    ├── __init__                                  // Preset parameters in the experiments.
│    ├── run_1_tasks.py                            // Run all 3 experiments for this work.
│    ├── run_2_packs.py                            // Package all the presented data from the experimental results.
│    ├── show_main.py                              // Paint figures (in the main text) from the generated data.
│    ├── show_supp.py                              // Paint figures (in the supporting materials) from the generated data.
│    ├── show_video.py                             // Create videos (in the supporting materials) from the generated data.
├── LICENSE                                        // License of this library (GPL-3.0 license).
├── README.md                                      // Description document of this library.
├── requirements.txt                               // Basic library requirements of this library.
```

## Citation
If you think this repository helps or being used in your research, please consider refer this work.
Here is a Bibtex entry:

````
@article{zhang2023representational,
  title={Representational capacity and numerical stability of network motifs and their role in optimizing artificial neural networks},
  author={Zhang, Haoling and Yang, Chao-Han Huck and Zenil, Hector and Chen, Pin-Yu and Shen, Yue and Tegner, Jesper},
  journal={TODO},
  year={2023}
}
````
