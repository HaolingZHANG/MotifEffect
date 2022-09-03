from datetime import datetime
from neat import Population, reporting, statistics
from neat import genome, stagnation, reproduction, species, config
from neat.six_util import iteritems, itervalues
from numpy import zeros

from practice.evolve import GSReproduction, NSReproduction, AdjustedReproduction, GlobalGenome, UpdatedSpeciesSet
from practice.motif import count_motifs_from_matrices, GraphType


class Monitor(object):

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def output(self, current_state, total_state, extra=None):
        """
        Output the current state of process.

        :param current_state: current state of process.
        :type current_state: int

        :param total_state: total state of process.
        :type total_state: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if self.last_time is None:
            self.last_time = datetime.now()

        if current_state == 0:
            return

        position = int(current_state / total_state * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (total_state - current_state) / current_state)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(total_state)) - len(str(current_state))) + str(current_state) + "/" + str(total_state)

        if current_state < total_state:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if current_state >= total_state:
            self.last_time = None
            print()


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

    :return: best genome (DefaultGenome), and additional reporters if required.
    :rtype: neat.genome.DefaultGenome or (neat.genome.DefaultGenome, list)
    """
    # In neat-python package, because of the internal damage (for evolving) to the key,
    # sometimes, geometry-based strategy can obtain KeyError.
    try:
        population = Population(model_config, initial_state)
        if need_stdout:
            population.add_reporter(reporting.StdOutReporter(False))
        population.add_reporter(statistics.StatisticsReporter())
        if additional_reporters is not None:
            for additional_reporter in additional_reporters:
                population.add_reporter(additional_reporter)
        best_genome = population.run(task.genomes_fitness, task.maximum_generation)
    except KeyError:
        return None

    if additional_reporters is None:
        return best_genome
    else:
        return best_genome, additional_reporters


def create_agent_config(path):
    """
    Create training configure of agent.

    :param path: configure path.
    :type path: str

    :return: agent configure.
    :rtype: neat.config.Config
    """
    if path.split(".")[-1] == "baseline":
        return config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                             species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                             path)
    elif path.split(".")[-1] == "geometry":
        return config.Config(GlobalGenome, GSReproduction,
                             UpdatedSpeciesSet, stagnation.DefaultStagnation,
                             path)
    elif path.split(".")[-1] == "novelty":
        return config.Config(genome.DefaultGenome, NSReproduction,
                             species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                             path)
    elif path.split(".")[-1] == "adjusted":
        return config.Config(genome.DefaultGenome, AdjustedReproduction,
                             species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                             path)
    else:
        raise ValueError("No such evolutionary strategy!")


def calculate_matrix_from_agent(agent, need_mapping=False):
    """
    Calculate the adjacency matrix of NEAT agent.

    :param agent: NEAT agent.
    :type agent: practice.evolve.NEATAgent

    :param need_mapping: need return the mapping between row index and node key.
    :type need_mapping: bool

    :return: adjacency matrix (and mapping between row index and node key is require).
    :rtype: tuple or numpy.ndarray
    """
    model_genome, neat_config = agent.get_genome(), agent.get_config()
    scale = neat_config.genome_config.num_inputs + neat_config.genome_config.num_outputs + len(model_genome.nodes)

    mapping, matrix = {}, zeros(shape=(scale, scale))
    for index in range(neat_config.genome_config.num_inputs):
        mapping[index - neat_config.genome_config.num_inputs] = index

    # add node key.
    index = neat_config.genome_config.num_inputs
    for node_key, node_gene in iteritems(model_genome.nodes):
        mapping[node_key] = index
        index += 1

    # add connect weight.
    for connect_gene in itervalues(model_genome.connections):
        if mapping.get(connect_gene.key[0]) is not None and mapping.get(connect_gene.key[1]) is not None:
            row = mapping.get(connect_gene.key[0])
            col = mapping.get(connect_gene.key[1])
            matrix[row, col] = connect_gene.weight

    if need_mapping:
        reversed_mapping = {}
        for key, value in mapping.items():
            reversed_mapping[value] = key
        return matrix, reversed_mapping
    else:
        return matrix


def calculate_agent_frequency(agents, reference_motifs, search_size=3, graph_type=GraphType.pn, pruning=True):
    """
    Calculate the motif frequency of NEAT agents.

    :param agents: NEAT agents.
    :type agents: list

    :param reference_motifs: reference motifs for order.
    :type reference_motifs: numpy.ndarray

    :param search_size: motif search size.
    :type search_size: int

    :param graph_type: type of graph.
    :type graph_type: grace.motif.GraphType

    :param pruning: pruning the arrows by box plot.
    :type pruning: bool

    :return: count collector for motif frequency.
    :rtype: dict
    """
    matrix_groups = []
    for agent in agents:
        matrix_groups.append(calculate_matrix_from_agent(agent, need_mapping=False))

    return count_motifs_from_matrices(matrices=matrix_groups, search_size=search_size, graph_type=graph_type,
                                      pruning=pruning, reference_motifs=reference_motifs)
