from copy import deepcopy
from neat import Population, reporting, statistics
from neat import genome, stagnation, reproduction, species, config
from neat.nn import FeedForwardNetwork, RecurrentNetwork

from practice.evolve import AdjustedGenome, AdjustedReproduction, create_adjacency_matrix
from practice.motif import count_motifs_from_adjacency_matrix


class DefaultAgent(object):

    def __init__(self, trained_model, description):
        """
        Initialize the agent.

        :param trained_model: trained model.
        :type trained_model: object

        :param description: description of this trained model.
        :type description: str
        """
        self.trained_model, self.description = trained_model, description

    def work(self, inputs):
        raise NotImplementedError("\"agent_do\" interface needs to be implemented.")


class NEATAgent(DefaultAgent):

    def __init__(self, model_genome, neat_config, description, action_handle):
        """
        Initialize the NEAT agent.

        :param model_genome: genome of NEAT model.
        :type model_genome: neat.genome.DefaultGenome

        :param neat_config: configure of NEAT algorithm.
        :type neat_config: neat.config.Config

        :param description: agent description.
        :type description: str

        :param action_handle: handle of action nodes.
        :type action_handle: function
        """
        if neat_config.genome_config.feed_forward:
            trained_model = FeedForwardNetwork.create(model_genome, neat_config)
        else:
            trained_model = RecurrentNetwork.create(model_genome, neat_config)
        super().__init__(trained_model, description)
        self.model_genome, self.neat_config = model_genome, neat_config
        self.action_handle = action_handle

    def work(self, inputs):
        """
        Run the agent through the inputted information.

        :param inputs: inputted information.
        :type inputs: numpy.ndarray

        :return: outputted information.
        :rtype: numpy.ndarray
        """
        # noinspection PyUnresolvedReferences
        outputs = self.trained_model.activate(inputs)
        action = self.action_handle(outputs)

        return action, outputs

    def get_fitness(self):
        """
        Get the fitness of agent.

        :return: fitness value.
        :rtype: float
        """
        return deepcopy(self.model_genome.fitness)

    def get_genome(self):
        """
        Get the genome information.

        :return: genome information.
        :rtype: neat.genome.DefaultGenome
        """
        return deepcopy(self.model_genome)

    def get_config(self):
        """
        Get the genome config.

        :return: genome config.
        :rtype: neat.config.Config
        """
        return deepcopy(self.neat_config)

    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix from the agent.

        :return: adjacency matrix.
        :rtype: numpy.ndarray
        """
        return create_adjacency_matrix(self.model_genome, self.neat_config.genome_config)

    def get_motif_counts(self, search_size=3, reference_motifs=None):
        """
        Get motif counts from the agent.

        :param search_size: size of search.
        :type search_size: int

        :param reference_motifs: reference motifs for order.
        :type reference_motifs: numpy.ndarray or list

        :return: motif counts.
        :rtype numpy.ndarray
        """
        return count_motifs_from_adjacency_matrix(self.get_adjacency_matrix(), search_size, reference_motifs)


def create_agent_config(config_path):
    """
    Create training configure of agent.

    :param config_path: configure path.
    :type config_path: str

    :return: agent configure.
    :rtype: neat.config.Config
    """
    if "baseline" in config_path.split(".")[-2]:
        return config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                             species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                             config_path)
    elif "adjusted" in config_path.split(".")[-2]:
        return config.Config(AdjustedGenome, AdjustedReproduction,
                             species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                             config_path)
    else:
        raise ValueError("No such evolutionary strategy!")


def obtain_best(task, model_config, need_stdout=False, additional_reporters=None, initial_state=None):
    """
    Obtain best genome in the specific task.

    :param task: used task.
    :type task: grace1.tasks.NEATGymTask

    :param model_config: genome config.
    :type model_config: neat.config.Config

    :param need_stdout: need standard output stream.
    :type need_stdout: bool

    :param additional_reporters: additional reporter.
    :type additional_reporters: list

    :param initial_state: initial state if required.
    :type initial_state: tuple

    :return: best genome, and additional reporters if required.
    :rtype: neat.genome.DefaultGenome or (neat.genome.DefaultGenome, list)
    """
    population = Population(model_config, initial_state)
    if need_stdout:
        population.add_reporter(reporting.StdOutReporter(False))
    population.add_reporter(statistics.StatisticsReporter())
    if additional_reporters is not None:
        for additional_reporter in additional_reporters:
            population.add_reporter(additional_reporter)
    best_genome = population.run(task.genomes_fitness, task.maximum_generation)

    if additional_reporters is None:
        return best_genome
    else:
        return best_genome, additional_reporters


def train_and_test(task, agent_name, agent_config, repeats, train_noise_generator, test_noise_generators):
    """
    Train and test the agents in a given NEAT task.

    :param task: task to train agents.
    :type task: practice.task.NEATCartPoleTask

    :param agent_name: name of trained agent.
    :type agent_name: str

    :param agent_config: agent configure.
    :type agent_config: neat.config.Config

    :param repeats: sample repeats.
    :type repeats: int

    :param train_noise_generator: noise generator for training process.
    :type train_noise_generator: practice.task.NormNoiseGenerator

    :param test_noise_generators: noise generators for evaluating process.
    :type test_noise_generators: dict

    :return: records.
    :rtype: list
    """
    records = []
    for _ in range(repeats):
        task.reset_experiences()
        task.set_noise(train_noise_generator)
        best_agent = NEATAgent(obtain_best(task, agent_config, need_stdout=True),
                               agent_config, agent_name, task.action_handle)
        experience = task.get_experiences()
        test_record = {}
        for label, test_noise_generator in test_noise_generators.items():
            task.set_noise(test_noise_generator)
            test_record[label] = task.calculate_fitness(task.run(best_agent)["rewards"])

        records.append((best_agent, experience, test_record))

    return records
