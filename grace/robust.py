from copy import deepcopy
from enum import Enum
from numpy import array, arange, zeros, random, repeat, expand_dims, linalg
from numpy import argmin, argmax, abs, min, max, sum, all, clip, inf

from grace.handle import Monitor


class NormNoiseGenerator(object):

    def __init__(self, norm_type, noise_level=1.0, noise_scale=1.0):
        """
        Initialize the noise iterator based on norm.

        :param norm_type: perturbed norm type in the noise iterator.
        :type norm_type: str

        :param noise_level: proportion of samples with noise in all samples.
        :type noise_level: float

        :param noise_scale: degree of noise attenuation.
        :type noise_scale: float
        """
        self.norm_type = norm_type
        self.noise_level = noise_level
        self.noise_scale = noise_scale

    # noinspection PyArgumentList
    def get_samples(self, sample, count, minimum_bounds, maximum_bounds):
        """
        Get noise samples based on the noise-free samples.

        :param sample: one noise-free sample.
        :type sample: numpy.ndarray

        :param count: noise sample number.
        :type count: int

        :param minimum_bounds: minimum bounds of observations.
        :type minimum_bounds: numpy.ndarray

        :param maximum_bounds: maximum bounds of observations.
        :type maximum_bounds: numpy.ndarray

        :return: noise samples, one or more.
        :rtype: numpy.ndarray
        """
        assert len(sample.shape) == 1

        if self.noise_scale == 0.0 or self.noise_level == 0.0:
            return sample

        if self.norm_type == "L-1":  # using Laplacian distribution.
            noises = random.laplace(size=(count, sample.shape[0]))
            normalized_noises = ((noises - min(noises)) / (max(noises) - min(noises)) - 0.5) * 2.0
        elif self.norm_type == "L-2":  # using Gaussian distribution.
            noises = random.normal(size=(count, sample.shape[0]))
            normalized_noises = ((noises - min(noises)) / (max(noises) - min(noises)) - 0.5) * 2.0
        elif self.norm_type == "L-inf":  # using uniform distribution.
            normalized_noises = random.uniform(low=-1.0, high=1.0, size=(count, sample.shape[0]))
        else:
            raise ValueError("No such norm type!")

        actual_noises = normalized_noises * self.noise_scale

        if self.noise_level < 1.0:
            if (1.0 - self.noise_level) * count > 1:
                noise_indices = arange(count)
                random.shuffle(noise_indices)
                ignored_indices = noise_indices[:int((1.0 - self.noise_level) * count)]
                actual_noises[ignored_indices] = 0.0
            else:
                for index in range(count):
                    if random.random() > self.noise_level:
                        actual_noises[index] = 0.0

        noise_samples = repeat(expand_dims(sample, axis=0), count, axis=0) + actual_noises

        for variable_index, (minimum_bound, maximum_bound) in enumerate(zip(minimum_bounds, maximum_bounds)):
            noise_samples[:, variable_index] = clip(noise_samples[:, variable_index], minimum_bound, maximum_bound)

        return noise_samples if count > 1 else noise_samples[0]


# noinspection PyTypeChecker
def intervene(task, agent, noise_scales, reward_calculator, random_seed=None, verbose=False):
    """
    Intervene the gym environment in the test process.

    :param task: available task.
    :type task: grace.task.GymTask

    :param agent: available agent.
    :type agent: grace.agent.DefaultAgent

    :param noise_scales: degrees of noise attenuation.
    :type noise_scales: numpy.ndarray or list

    :param reward_calculator: calculator of obtained rewards, like sum, mean, or others.
    :type reward_calculator: function

    :param random_seed: random seed for initializing the environment and creating the noise.
    :type random_seed: int or None

    :param verbose: need to show process log.
    :type verbose: bool

    :return: reward record.
    :rtype: numpy.ndarray
    """
    test_records = zeros(shape=(len(noise_scales), task.iterations))

    if random_seed is None:
        random.seed(random_seed)

    for index_1, noise_scale in enumerate(noise_scales):
        task.noise_generator.noise_scale = noise_scale
        reward_collector = task.run(agent=agent)["rewards"]
        for index_2, rewards in enumerate(reward_collector):
            test_records[index_1, index_2] = reward_calculator(rewards)

    if random_seed is None:
        random.seed(None)

    return test_records


# noinspection PyTypeChecker
def calculate_grace(task, agent, noise_generator, noise_sampling=20, state_sampling=20, replay=False,
                    random_seed=None, verbose=False):
    """
    Calculate Grace scores.

    :param task: available task.
    :type task: grace.task.GymTask

    :param agent: available agent.
    :type agent: grace.agent.DefaultAgent

    :param noise_generator: noise generator for calculating robustness score.
    :type noise_generator: grace.robust.NormNoiseGenerator

    :param noise_sampling: number of noise sample.
    :type noise_sampling: int

    :param state_sampling: number of state (per variable in the state) when "replay" is false.
    :type state_sampling: int

    :param replay: play in the environment through agents to create states.
    :type replay: bool

    :param random_seed: random seed for initializing the environment and creating the noise.
    :type random_seed: int or None

    :param verbose: need to show process log.
    :type verbose: bool

    :return: GRACE scores.
    :rtype: numpy.ndarray
    """

    if random_seed is None:
        random.seed(random_seed)

    if replay:
        saved_states = task.run_1_iteration(agent=agent)["states"]
    else:
        saved_states = task.sampling_states(sampling=state_sampling)

    scores, total, monitor = [], len(saved_states), Monitor()
    norm_value = {"L-1": inf, "L-2": 2, "L-inf": 1}[noise_generator.norm_type]
    minimum_bounds, maximum_bounds = task.get_state_range()
    for current, saved_state in enumerate(saved_states):
        # obtain the action by the selected agent from the saved observation.
        action_values = task.run_1_step(agent=agent, state=saved_state, reset=True)["action"]

        # calculate the numerator, f_max(state) - f_min(state)
        numerator = max(action_values) - min(action_values)

        # generate noise samples in required norm based on the saved observation.
        noise_samples = noise_generator.get_samples(sample=saved_state, count=noise_sampling,
                                                    minimum_bounds=minimum_bounds, maximum_bounds=maximum_bounds)

        def calculate_handle(positive_h_values, negative_h_values):
            """
            Calculate gradient from a special strategy.

            :param positive_h_values: positive perturbed actions.
            :type positive_h_values: numpy.ndarray

            :param negative_h_values: negative perturbed actions.
            :type negative_h_values: numpy.ndarray

            :return: gradient of special handle.
            :rtype: numpy.ndarray
            """
            positive_difference = positive_h_values[argmax(action_values)] - positive_h_values[argmin(action_values)]
            negative_difference = negative_h_values[argmax(action_values)] - negative_h_values[argmin(action_values)]
            return positive_difference - negative_difference

        denominators = []
        for state in noise_samples:
            # obtain a perturbed estimated gradient based on the noise sample for further calculation.
            denominators.append(task.estimate_gradient(agent=agent, state=state, calculate_handle=calculate_handle))

        scores.append(numerator / max(linalg.norm(array(denominators), ord=norm_value, axis=1)))

        if verbose:
            monitor.output(current + 1, total, extra={"score": scores[-1], "state": saved_state})

    if random_seed is None:
        random.seed(None)

    return array(scores)
