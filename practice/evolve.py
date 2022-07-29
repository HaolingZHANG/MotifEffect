from copy import deepcopy
from math import sqrt
from neat import DefaultGenome, DefaultSpeciesSet, DefaultReproduction
from neat.config import DefaultClassConfig, ConfigParameter
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig
from neat.math_util import mean, stdev
from neat.six_util import iteritems, itervalues, iterkeys
from neat.species import Species, GenomeDistanceCache
from numpy import array, min, max, argsort, ceil, isnan
from pandas import Series
from random import choice, sample, randint
from sklearn.cluster import KMeans, SpectralClustering, Birch


class BiReproduction(DefaultReproduction):

    def __init__(self, train_config, train_reporters, train_stagnation):
        super().__init__(train_config, train_reporters, train_stagnation)
        self.best_genome = None
        self.genome_config = None
        self.genome_type = None
        self.global_rate = None

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter("init_distance", float, 5),
                                   ConfigParameter("min_distance", float, 0.2),
                                   ConfigParameter("correlation_rate", float, -0.5),
                                   ConfigParameter("search_count", int, 1),
                                   ConfigParameter("cluster_method", str, "kmeans++")])

    def create_new(self, genome_type, genome_config, num_genomes):
        if genome_config.num_hidden + genome_config.num_inputs + genome_config.num_outputs > genome_config.max_node_num:
            raise Exception("config: max_node_num must larger than num_inputs + num_outputs + num_hidden")

        self.genome_config = genome_config
        self.genome_type = genome_type

        new_genomes = {}
        distance_matrix = [[float("inf") for _ in range(num_genomes - 1)] for _ in range(num_genomes - 1)]

        for created_index in range(num_genomes):
            key = next(self.genome_indexer)
            new_genome = genome_type(key)
            count = 0
            while True:
                new_genome.configure_new(genome_config)
                min_distance = float("inf")
                for check_index, saved_genome in new_genomes.items():
                    # noinspection PyUnresolvedReferences
                    current_distance = saved_genome.distance(new_genome, genome_config)
                    distance_matrix[created_index - 1][check_index - 1] = current_distance
                    distance_matrix[check_index - 1][created_index - 1] = current_distance
                    if min_distance > current_distance:
                        min_distance = current_distance
                if min_distance >= self.reproduction_config.init_distance:
                    break

                count += 1

                if count > self.reproduction_config.search_count:
                    raise Exception("\"init_distance\" is too large for the whole landscape," +
                                    "please reduce init_distance or try again!")

            new_genomes[key] = new_genome
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, used_config, used_species, pop_size, generation):
        # obtain current genomes and current evolution speed.
        genome_clusters, cluster_centers = self.obtain_clusters(used_species, pop_size)

        # obtain topological genomes and near genomes.
        new_genomes = self.obtain_phenotypic_network(pop_size, genome_clusters, cluster_centers)

        # aggregate final population
        new_population = {}
        for index, new_genome in enumerate(new_genomes):
            new_genome.key = index
            new_population[index] = new_genome

        return new_population

    def obtain_clusters(self, used_species, pop_size):
        # obtain all genomes from species.
        current_genomes = []
        for i, value in used_species.species.items():
            members = value.members
            for key, individual in members.items():
                current_genomes.append(individual)

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda g: g.fitness)

        # calculate speed list and avg adjusted fitness
        avg_adjusted_fitness = 0

        if len(current_genomes) > pop_size:
            feature_matrices = []
            for current_genome in current_genomes:
                feature_matrices.append([])
                for feature_slice in current_genome.feature_matrix:
                    feature_matrices[-1] += deepcopy(list(feature_slice))

            # cluster the current network based on the size of population.
            labels, centers = self.cluster(feature_matrices, pop_size, len(current_genomes))

            genome_clusters = [[] for _ in range(pop_size)]
            for index, cluster_index in enumerate(labels):
                genome_clusters[cluster_index].append(current_genomes[index])

            for genome_cluster in genome_clusters:
                avg_adjusted_fitness += genome_cluster[0].fitness / pop_size

            self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

            return genome_clusters, centers
        else:
            genome_clusters = []
            for current_genome in current_genomes:
                genome_clusters.append([current_genome])
                avg_adjusted_fitness += current_genome.fitness / pop_size

            self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

            return genome_clusters, None

    def obtain_phenotypic_network(self, pop_size, genome_clusters, cluster_centers):
        if cluster_centers is not None:
            saved_genomes, correlations = [], []

            # analyze the correlation between fitting degree and spatial position (negative correlation normally).
            for genome_cluster in genome_clusters:
                distances, fitnesses = [0], [genome_cluster[0].fitness]
                saved_genomes.append(genome_cluster[0])
                for index in range(1, len(genome_cluster)):
                    distances.append(genome_cluster[0].distance(genome_cluster[index], self.genome_config))
                    fitnesses.append(genome_cluster[index].fitness)

                if len(fitnesses) > 1:
                    correlations.append(round(Series(distances).corr(Series(fitnesses)), 2))
                else:
                    correlations.append(-1.00)

                for index in range(len(correlations)):
                    if isnan(correlations[index]):
                        correlations[index] = 0

            self.reporters.info("Correlations: " + str(correlations))

            new_genomes = []
            # construct the topology of the phenotypical network
            for index_1 in range(pop_size):
                cluster_1 = genome_clusters[index_1]
                for index_2 in range(index_1 + 1, pop_size):
                    cluster_2 = genome_clusters[index_2]

                    if distance_between_two_matrices(cluster_1[0].feature_matrix, cluster_2[0].feature_matrix) \
                            > self.reproduction_config.min_distance:

                        # If the two clusters both have highly correlations,
                        # it means that the current network of these two clusters has a better description of phenotype,
                        # and then evolution is carried out according to the original method.
                        if correlations[index_1] >= self.reproduction_config.correlation_rate \
                                and correlations[index_2] >= self.reproduction_config.correlation_rate:
                            topo_genome = self.obtain_global_genome(cluster_centers[index_1],
                                                                    cluster_centers[index_2],
                                                                    saved_genomes + new_genomes, -1)
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome = self.obtain_near_genome(cluster_1[0],
                                                                      saved_genomes + new_genomes + cluster_1, -1)
                            else:
                                near_genome = self.obtain_near_genome(cluster_2[0],
                                                                      saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome is not None:
                                new_genomes.append(near_genome)
                            if topo_genome is not None:
                                new_genomes.append(topo_genome)

                        elif correlations[index_1] >= self.reproduction_config.correlation_rate > correlations[index_2]:
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)
                            else:
                                near_genome_1 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)

                        elif correlations[index_2] >= self.reproduction_config.correlation_rate > correlations[index_1]:
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)

                            else:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)
                        else:
                            near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                    saved_genomes + new_genomes + cluster_1, -1)
                            near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                    saved_genomes + new_genomes + cluster_2, -1)
                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)

            new_genomes += saved_genomes
        else:
            # create the initial topology network (binary search & near search).
            new_genomes = []
            for genome_cluster in genome_clusters:
                new_genomes.append(genome_cluster[0])

            for index_1 in range(pop_size):
                genome_1 = new_genomes[index_1]
                for index_2 in range(index_1 + 1, pop_size):
                    genome_2 = new_genomes[index_2]

                    if genome_1.distance(genome_2, self.genome_config) > self.reproduction_config.min_distance:
                        # add near genome (limit search count)
                        near_genome = self.obtain_near_genome(genome_1, new_genomes, -1)
                        # add center genome
                        topo_genome = self.obtain_global_genome(genome_1.feature_matrix, genome_2.feature_matrix,
                                                                new_genomes, -1)
                        if near_genome is not None:
                            new_genomes.append(near_genome)
                        if topo_genome is not None:
                            new_genomes.append(topo_genome)

        return new_genomes

    def cluster(self, feature_matrices, pop_size, iteration):
        centers = []
        if self.reproduction_config.cluster_method == "kmeans++":
            method = KMeans(n_clusters=pop_size, max_iter=iteration)
        elif self.reproduction_config.cluster_method == "spectral":
            method = SpectralClustering(n_clusters=pop_size)
        elif self.reproduction_config.cluster_method == "birch":
            method = Birch(n_clusters=pop_size)
        else:
            method = KMeans(n_clusters=pop_size, max_iter=iteration, init="random")

        method.fit(feature_matrices)
        for cluster_center in method.cluster_centers_:
            feature_matrix = []
            for index in range(self.genome_config.max_node_num):
                feature_matrix.append(list(cluster_center[index * self.genome_config.max_node_num:
                                                          (index + 1) * self.genome_config.max_node_num + 1]))
            centers.append(feature_matrix)

        return method.labels_, centers

    def obtain_global_genome(self, matrix_1, matrix_2, saved_genomes, index):
        center_genome = create_center_new(matrix_1, matrix_2, self.genome_config, index)
        is_input = True
        for check_genome in saved_genomes:
            if center_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                is_input = False

        if is_input:
            return center_genome

        return None

    def obtain_near_genome(self, parent_genome, saved_genomes, index):
        count = 0
        while count < self.reproduction_config.search_count:
            near_genome = create_near_new(parent_genome, self.genome_config, index)
            is_input = True
            for check_genome in saved_genomes:
                if near_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                    is_input = False
            if is_input:
                return near_genome

            count += 1

        return None


class GSReproduction(BiReproduction):

    def obtain_global_genome(self, matrix_1, matrix_2, saved_genomes, index):
        center_genome = create_golden_section_new(matrix_1, matrix_2, self.genome_config, index)
        is_input = True
        for check_genome in saved_genomes:
            if center_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                is_input = False

        if is_input:
            return center_genome

        return None


class NSReproduction(DefaultReproduction):

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter("elitism", int, 0),
                                   ConfigParameter("survival_threshold", float, 0.2),
                                   ConfigParameter("min_species_size", int, 2),
                                   ConfigParameter("neighbor_size", int, 2),
                                   ConfigParameter("novelty_rate", float, 0)])

    def reproduce(self, config, species, pop_size, generation):
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)

        if not remaining_species:
            species.species = {}
            return {}
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)

        fitness_range = max([1.0, max_fitness - min_fitness])
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size

        min_species_size = max([min_species_size, self.reproduction_config.elitism])
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}

        for spawn, s in zip(spawn_amounts, remaining_species):
            spawn = max([spawn, self.reproduction_config.elitism])

            assert spawn > 0

            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            n_scores, f_scores = [], []
            for index_1 in range(len(old_members)):
                f_scores.append(old_members[index_1][1].fitness)
                ds = []
                for index_2 in range(len(old_members)):
                    if index_1 != index_2:
                        ds.append(old_members[index_1][1].distance(old_members[index_2][1], config.genome_config))
                ds = array(ds)
                ds.sort()
                if len(ds) > 0:
                    n_scores.append(mean(array(ds[: self.reproduction_config.neighbor_size])))
                else:
                    n_scores.append(-1)

            for index in range(len(n_scores)):
                if n_scores[index] == -1:
                    n_scores[index] = max(n_scores)

            n_scores, f_scores = array(n_scores), array(f_scores)
            f_scores -= min(f_scores)

            if config.reproduction_config.novelty_rate == 1:
                mixed_scores = n_scores / max(n_scores)
            elif config.reproduction_config.novelty_rate == 0.0:
                mixed_scores = f_scores / max(f_scores)
            else:
                n_scores = (n_scores / max(n_scores)) * config.reproduction_config.novelty_rate
                f_scores = (f_scores / max(f_scores)) * (1 - config.reproduction_config.novelty_rate)
                mixed_scores = n_scores + f_scores

            sorted_members = []
            for index in argsort(-mixed_scores):
                sorted_members.append(old_members[index])

            old_members = sorted_members

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

                parent1_id, parent1 = choice(old_members)
                parent2_id, parent2 = choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child

        return new_population


class GlobalGenome(DefaultGenome):

    @classmethod
    def parse_config(cls, param_dict):
        super().parse_config(param_dict)
        return GlobalGenomeConfig(param_dict)

    def __init__(self, key):
        super().__init__(key)
        self.feature_matrix = None

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def configure_new(self, config):
        """
        Create new genome by configure, and then create the feature matrix.

        :param config: genome config.
        :type config: grace1.neats.entity.GlobalGenomeConfig
        """
        # create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # add hidden nodes if requested.
        if config.num_hidden > 0:
            for node_key in range(len(config.output_keys), config.num_hidden + len(config.output_keys)):
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # add connections with global random.
        for input_id, output_id in self.compute_connections(config):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

        # add feature matrix
        self.set_feature_matrix(config)

    @staticmethod
    def compute_connections(config):
        start = len(config.output_keys)
        stop = config.num_hidden + len(config.output_keys)
        hidden_keys = sample([index for index in range(start, stop)], randint(0, config.num_hidden))

        connections = []

        if len(hidden_keys) == 0:
            for input_id in config.input_keys:
                for output_id in config.output_keys:
                    if randint(0, 1):
                        connections.append((input_id, output_id))

            if len(connections) == 0:
                input_id = sample(config.input_keys, 1)
                for output_id in config.output_keys:
                    connections.append((input_id[0], output_id))
        else:
            chosen_keys = set()

            # from input and hidden nodes to hidden nodes.
            for index in range(len(hidden_keys)):
                before_keys = config.input_keys + hidden_keys[:index]
                for in_degree in sample(before_keys, randint(1, len(before_keys))):
                    connections.append((in_degree, hidden_keys[index]))
                    chosen_keys.add(in_degree)
                    chosen_keys.add(hidden_keys[index])

            # from input and hidden nodes to output nodes.
            chosen_keys = list(chosen_keys)
            for output_id in config.output_keys:
                for in_degree in sample(chosen_keys, randint(1, len(chosen_keys))):
                    connections.append((in_degree, output_id))

        connections = sorted(connections)

        return connections

    def feature_matrix_new(self, feature_matrix, config):
        """
        Create new genome by feature matrix.

        :param feature_matrix: obtained feature matrix.
        :type feature_matrix: list

        :param config: genome config.
        :type config: practice.agent.GlobalGenomeConfig
        """
        self.feature_matrix = feature_matrix

        # create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # add hidden nodes by feature matrix if requested.
        for node_key in range(config.num_hidden):
            node = DefaultNodeGene(node_key)
            node.bias = feature_matrix[node_key + config.num_inputs][0]
            node.response = config.response_init_mean
            node.activation = config.activation_default
            node.aggregation = config.aggregation_default
            self.nodes[node_key] = node

        # set connections by feature matrix.
        for in_index in range(len(feature_matrix)):
            for out_index in range(1, len(feature_matrix[in_index])):
                if feature_matrix[in_index][out_index] > 0:
                    connection = DefaultConnectionGene((in_index, out_index - 1))
                    connection.weight = feature_matrix[in_index][out_index]
                    connection.enabled = config.enabled_default
                    self.connections[connection.key] = connection

    # noinspection PyUnresolvedReferences
    def set_feature_matrix(self, config):
        """
        Set the feature matrix for this genome.

        :param config: genome config.
        :type config: grace.evolve.GlobalGenomeConfig
        """
        # bia + weight
        self.feature_matrix = [[0 for _ in range(config.max_node_num + 1)] for _ in range(config.max_node_num)]

        # position mapping of feature matrix
        mapping = {}
        for index in range(config.num_inputs):
            mapping[index - config.num_inputs] = index

        # add node bias
        index = config.num_inputs
        for node_key, node_gene in iteritems(self.nodes):
            self.feature_matrix[index][0] = node_gene.bias
            mapping[node_key] = index
            index += 1

        # add connect weight
        for connect_gene in itervalues(self.connections):
            if mapping.get(connect_gene.key[0]) is not None and mapping.get(connect_gene.key[1]) is not None:
                row = mapping.get(connect_gene.key[0])
                col = mapping.get(connect_gene.key[1]) + 1
                self.feature_matrix[row][col] = connect_gene.weight

    def distance(self, other, config):
        """
        Obtain distance by two feature matrix.

        :param other: another genome.
        :type other: practice.agent.GlobalGenome

        :param config: genome config.
        :type config: practice.agent.GlobalGenomeConfig

        :return: distance of two genomes.
        """
        if other.feature_matrix is None:
            other.set_feature_matrix(config)

        return distance_between_two_matrices(self.feature_matrix, other.feature_matrix)

    # noinspection PyUnresolvedReferences
    def mutate_add_node(self, config):
        """
        Mutate add node when current hidden node (when node number less than the node range).

        :param config:genome config.
        :type config: practice.agent.GlobalGenomeConfig
        """
        if config.max_node_num - config.num_inputs - config.num_outputs > len(self.nodes):
            super().mutate_add_node(config)

    def __str__(self):
        s = super().__str__()
        s += "\nFeature Matrix:"
        for row in self.feature_matrix:
            s += "\n\t" + str(row)

        return s


# noinspection PyMissingConstructor
class GlobalGenomeConfig(DefaultGenomeConfig):

    def __init__(self, params):
        """
        Initialize config by params, add ConfigParameter("max_node_num", int)
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
                        ConfigParameter("max_node_num", int),
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
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None


class UpdatedSpeciesSet(DefaultSpeciesSet):

    def speciate(self, config, population, generation):
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            if len(candidates) > 0:
                ignored_rdist, new_rep = candidates[0]
                for d, r in candidates[1:]:
                    if d < ignored_rdist:
                        ignored_rdist, new_rep = d, r
                new_rid = new_rep.key
                new_representatives[sid] = new_rid
                new_members[sid] = [new_rid]
                unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if len(candidates) > 0:
                ignored_sdist, sid = candidates[0]
                for d, s in candidates[1:]:
                    if d < ignored_sdist:
                        ignored_sdist, sid = d, s
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info("Mean genetic distance {0:.3f}, standard deviation {1:.3f}".format(gdmean, gdstdev))


def create_center_new(feature_matrix_1, feature_matrix_2, config, key):
    """
    Create a new genome at the midpoint of two genomes.

    :param feature_matrix_1: feature_matrix in one genome.
    :type feature_matrix_1: list

    :param feature_matrix_2: feature_matrix in another genome.
    :type feature_matrix_2: list

    :param config: genome config.
    :type config: grace.evolve.GlobalGenomeConfig

    :param key: key of the new genome.
    :type key: int

    :return: center genome.
    :rtype: grace.evolve.GlobalGenome
    """
    new_feature_matrix = []
    for row_1, row_2 in zip(feature_matrix_1, feature_matrix_2):
        new_row = []
        for value_1, value_2 in zip(row_1, row_2):
            new_row.append((value_1 + value_2) / 2.0)

        new_feature_matrix.append(new_row)

    new_genome = GlobalGenome(key)
    new_genome.feature_matrix_new(new_feature_matrix, config)

    return new_genome


def create_golden_section_new(feature_matrix_1, feature_matrix_2, config, key):
    """
    Create a new genome at the golden-section of two genomes, near genome 1.

    :param feature_matrix_1: feature_matrix in one genome.
    :type feature_matrix_1: list

    :param feature_matrix_2: feature_matrix in another genome.
    :type feature_matrix_2: list

    :param config: genome config.
    :type config: grace.evolve.GlobalGenomeConfig

    :param key: key of the new genome.
    :type key: int

    :return: center genome.
    :rtype: grace.evolve.GlobalGenome
    """
    new_feature_matrix = []
    for row_1, row_2 in zip(feature_matrix_1, feature_matrix_2):
        new_row = []
        for value_1, value_2 in zip(row_1, row_2):
            new_row.append(value_1 + (value_2 - value_1) * ((3 - sqrt(5)) / 2))

        new_feature_matrix.append(new_row)

    new_genome = GlobalGenome(key)
    new_genome.feature_matrix_new(new_feature_matrix, config)

    return new_genome


def create_near_new(genome, config, key):
    """
    Create a new genome near the old genome.

    :param genome: original genome.
    :type genome: grace.evolve.GlobalGenome

    :param config: genome config.
    :type config: grace.evolve.GlobalGenomeConfig

    :param key: key of new genome.
    :type key: int

    :return: the new genome.
    :rtype: grace.evolve.GlobalGenome
    """
    new_genome = deepcopy(genome)
    new_genome.key = key
    new_genome.fitness = None
    new_genome.mutate(config)
    new_genome.set_feature_matrix(config)

    return new_genome


def distance_between_two_matrices(matrix_1, matrix_2):
    """
    Obtain the distance between two matrices.

    :param matrix_1: one matrix.
    :type matrix_1: list

    :param matrix_2: another matrix.
    :type matrix_2: list

    :return: the distance.
    :rtype: float
    """
    distance = 0
    for row_1, row_2 in zip(matrix_1, matrix_2):
        for value_1, value_2 in zip(row_1, row_2):
            distance += pow(value_1 - value_2, 2)

    return sqrt(distance)