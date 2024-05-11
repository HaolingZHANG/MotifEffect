"""
@Author      : Haoling Zhang
@Description : Construction of neuroevolution methods.
"""
# noinspection PyPackageRequirements
from neat import DefaultGenome, DefaultReproduction
# noinspection PyPackageRequirements
from neat.config import Config, ConfigParameter
# noinspection PyPackageRequirements
from neat.activations import ActivationFunctionSet
# noinspection PyPackageRequirements
from neat.aggregations import AggregationFunctionSet
# noinspection PyPackageRequirements
from neat.genome import DefaultGenomeConfig
# noinspection PyPackageRequirements
from neat.math_util import mean
# noinspection PyPackageRequirements
from neat.six_util import iteritems, itervalues
from numpy import ndarray, zeros, array, min, max, ceil
from random import choice
from typing import Union

from practice.motif import detect_motifs_from_adjacency_matrix


class AdjustedGenome(DefaultGenome):

    @classmethod
    def parse_config(cls,
                     param_dict: dict):
        # noinspection PyCompatibility
        super().parse_config(param_dict)
        return AdjustedGenomeConfig(param_dict)


class AdjustedGenomeConfig(DefaultGenomeConfig):

    # noinspection PyMissingConstructor
    def __init__(self,
                 params: dict):
        """
        Initialize config by params, add ConfigParameter("remove_type", str)

        :param params: parameters of adjusted genome.
        :type params: dict
        """
        # create full set of available activation functions.
        self.num_inputs = 0
        self.num_outputs = 0
        self.single_structural_mutation = None
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter("num_inputs", int),
                        ConfigParameter("num_outputs", int),
                        ConfigParameter("num_hidden", int),
                        ConfigParameter("remove_type", str),
                        ConfigParameter("feed_forward", bool),
                        ConfigParameter("compatibility_disjoint_coefficient", float),
                        ConfigParameter("compatibility_weight_coefficient", float),
                        ConfigParameter("conn_add_prob", float),
                        ConfigParameter("conn_delete_prob", float),
                        ConfigParameter("node_add_prob", float),
                        ConfigParameter("node_delete_prob", float),
                        ConfigParameter("single_structural_mutation", bool, "false"),
                        ConfigParameter("structural_mutation_surer", str, "default"),
                        ConfigParameter("initial_connection", str, "unconnected")]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params["node_gene_type"]
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params["connection_gene_type"]
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        if "partial" in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError("Partial connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in DefaultGenomeConfig.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        if self.structural_mutation_surer.lower() in ["1", "yes", "true", "on"]:
            self.structural_mutation_surer = "true"
        elif self.structural_mutation_surer.lower() in ["0", "no", "false", "off"]:
            self.structural_mutation_surer = "false"
        elif self.structural_mutation_surer.lower() == "default":
            self.structural_mutation_surer = "default"
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None


class AdjustedReproduction(DefaultReproduction):

    def create_new(self,
                   genome_type,
                   genome_config: Union[DefaultGenomeConfig, AdjustedGenomeConfig],
                   num_genomes: int):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            while True:  # add a forgoing check.
                genome = genome_type(key)
                genome.configure_new(genome_config)
                if not self.detect(genome, genome_config):  # screen special motifs.
                    break
            new_genomes[key] = genome
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self,
                  config: Config,
                  species,
                  pop_size: int,
                  generation: int):
        # Filter out stagnated species, collect the set of non-stagnated species members,
        # and compute their average adjusted fitness.
        all_fitnesses, remaining_species = [], []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}

        # Find minimum/maximum fitness across the entire population, for use in species adjusted fitness computation.
        min_fitness, max_fitness = min(all_fitnesses), max(all_fitnesses)
        fitness_range = max([1.0, max_fitness - min_fitness])
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        min_species_size = max([min_species_size, self.reproduction_config.elitism])
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max([spawn, self.reproduction_config.elitism])

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(ceil(self.reproduction_config.survival_threshold * len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max([repro_cutoff, 2])
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                while True:  # add a forgoing check.
                    parent1_id, parent1 = choice(old_members)
                    parent2_id, parent2 = choice(old_members)
                    child = config.genome_type(gid)
                    child.configure_crossover(parent1, parent2, config.genome_config)
                    child.mutate(config.genome_config)
                    if not self.detect(child, config.genome_config):  # screen special motifs.
                        break

                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population

    @staticmethod
    def detect(model_genome: Union[DefaultGenome, AdjustedGenome],
               genome_config: Union[DefaultGenomeConfig, AdjustedGenomeConfig]):
        """
        Detect selected motifs based on the adjacency matrix of agent.

        :param model_genome: model of genome.
        :type model_genome: neat.genome.DefaultGenome or practice.evolve.AdjustedGenome

        :param genome_config: genome configuration.
        :type genome_config: neat.genome.DefaultGenomeConfig or practice.evolve.AdjustedGenomeConfig

        :return: detection result.
        :rtype: bool
        """
        # To more rigorously prohibit the generation of selected motifs,
        # we do not consider the temporary variable "enabled" in connections during the training phase.
        adjacency_matrix = create_adjacency_matrix(model_genome, genome_config, consider_enable=False)

        # noinspection PyUnresolvedReferences
        if genome_config.remove_type in ["i", "a"]:
            detected_motifs = array([[[+0, -1, +1], [+0, +0, +1], [+0, +0, +0]],
                                     [[+0, +1, +1], [+0, +0, -1], [+0, +0, +0]],
                                     [[+0, +1, -1], [+0, +0, +1], [+0, +0, +0]],
                                     [[+0, -1, -1], [+0, +0, -1], [+0, +0, +0]]])
            if detect_motifs_from_adjacency_matrix(adjacency_matrix, 3, detected_motifs):
                return True

        # noinspection PyUnresolvedReferences
        if genome_config.remove_type in ["c", "a"]:
            detected_motifs = array([[[+0, +1, +1], [+0, +0, +1], [+0, +0, +0]],
                                     [[+0, -1, +1], [+0, +0, -1], [+0, +0, +0]],
                                     [[+0, -1, -1], [+0, +0, +1], [+0, +0, +0]],
                                     [[+0, +1, -1], [+0, +0, -1], [+0, +0, +0]]])
            if detect_motifs_from_adjacency_matrix(adjacency_matrix, 3, detected_motifs):
                return True

        return False


def create_adjacency_matrix(model_genome: Union[DefaultGenome, AdjustedGenome],
                            genome_config: Union[DefaultGenomeConfig, AdjustedGenomeConfig],
                            consider_enable: bool = True) \
        -> ndarray:
    """
    Create adjacency matrix based on the genome and its corresponding configuration.

    :param model_genome: model of genome.
    :type model_genome: neat.genome.DefaultGenome or practice.evolve.AdjustedGenome

    :param genome_config: genome configuration.
    :type genome_config: neat.genome.DefaultGenomeConfig or practice.evolve.AdjustedGenomeConfig

    :param consider_enable: only consider the weight of connection if it is enabled.
    :type consider_enable: bool

    :return: corresponding adjacency matrix.
    :rtype: numpy.ndarray
    """
    input_number, output_number = genome_config.num_inputs, genome_config.num_outputs
    scale = input_number + output_number + len(model_genome.nodes)

    mapping, reverse_mapping, adjacency_matrix = {}, {}, zeros(shape=(scale, scale))
    for index in range(genome_config.num_inputs):
        mapping[index - genome_config.num_inputs] = index
        reverse_mapping[index] = index - genome_config.num_inputs

    index = genome_config.num_inputs
    for node_key, node_gene in iteritems(model_genome.nodes):
        mapping[node_key] = index
        reverse_mapping[index] = node_key
        index += 1

    for connect_gene in itervalues(model_genome.connections):
        row, col = mapping[connect_gene.key[0]], mapping[connect_gene.key[1]]
        if consider_enable:
            if connect_gene.enabled:
                adjacency_matrix[row, col] = connect_gene.weight
        else:
            adjacency_matrix[row, col] = connect_gene.weight

    # remove connection between input nodes.
    adjacency_matrix[:input_number, :input_number] = 0

    # remove connection from output nodes to other nodes.
    adjacency_matrix[input_number:input_number + output_number, input_number + output_number:] = 0

    return adjacency_matrix
