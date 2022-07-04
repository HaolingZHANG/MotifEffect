from copy import deepcopy
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from numpy import expand_dims, squeeze


class DefaultAgent(object):

    def __init__(self, trained_model, description, extra_input_dim=0, extra_output_dim=0):
        """
        Initialize the agent.

        :param trained_model:
        :param description:
        :param extra_input_dim:
        :param extra_output_dim:
        """
        self.trained_model, self.description = trained_model, description
        self.extra_input_dim, self.extra_output_dim = extra_input_dim, extra_output_dim

    def work(self, inputs):
        return self.handle(self.agent_do(self.handle(inputs, self.extra_input_dim)), self.extra_output_dim)

    def agent_do(self, inputs):
        raise NotImplementedError("\"agent_do\" interface needs to be implemented.")

    @staticmethod
    def handle(data, extra_dim):
        if extra_dim > 0:
            for add_dim in range(extra_dim):
                data = expand_dims(data, 0)
        elif extra_dim < 0:
            for del_dim in range(abs(extra_dim)):
                data = squeeze(data, axis=0)

        return data


class NEATAgent(DefaultAgent):

    def __init__(self, model_genome, neat_config, description, action_handle=None,
                 extra_input_dim=0, extra_output_dim=0):
        """
        Initialize the NEAT agent.

        :param model_genome:
        :param neat_config:
        :param description:
        :param action_handle:
        :param extra_input_dim:
        :param extra_output_dim:
        """
        if neat_config.genome_config.feed_forward:
            trained_model = FeedForwardNetwork.create(model_genome, neat_config)
        else:
            trained_model = RecurrentNetwork.create(model_genome, neat_config)
        super().__init__(trained_model, description, extra_input_dim, extra_output_dim)
        self.model_genome, self.neat_config = model_genome, neat_config
        self.action_handle = action_handle

    def agent_do(self, inputs):
        """
        Run the agent through the inputted information.

        :param inputs: inputted information.
        :type inputs: numpy.ndarray

        :return: outputted information.
        :rtype: numpy.ndarray
        """
        outputs = self.trained_model.activate(inputs)

        if self.action_handle is not None:
            outputs = self.action_handle(outputs)

        return outputs

    def get_fitness(self):
        """
        Get the fitness of agent.

        :return: fitness value.
        :rtype: float
        """
        return self.model_genome.fitness

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
