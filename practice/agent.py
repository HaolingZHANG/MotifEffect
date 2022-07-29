from copy import deepcopy
from neat.nn import FeedForwardNetwork, RecurrentNetwork


class DefaultAgent(object):

    def __init__(self, trained_model, description):
        """
        Initialize the agent.

        :param trained_model:
        :param description:
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

        :param action_handle: handle of action array.
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
