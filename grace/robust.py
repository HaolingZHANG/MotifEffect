from enum import Enum
from numpy import array, arange, random, repeat, expand_dims, linalg, argmin, argmax, abs, min, max, sum, all, clip, inf

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
        assert sample.shape == 1

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


def calculate_grace(task, agent, noise_generator, sample_number, saved_states, verbose):
    """
    Calculate Grace scores.

    :param noise_generator: noise generator for calculating robustness score.
    :type noise_generator: grace.robust.NormNoiseGenerator

    :param sample_number: number of noise sample.
    :type sample_number: int

    :param saved_states: a collection of observations saved during the agent processing environment.
    :type saved_states: numpy.ndarray

    :param verbose: need to show process log.
    :type verbose: bool
    """
    numerators, denominators, total_steps, monitor = [], [], len(saved_states), Monitor()
    minimum_bounds, maximum_bounds = task.get_state_range()
    for current_step, saved_state in enumerate(saved_states):
        # obtain the action by the selected agent from the saved observation.
        action = task.run_1_step(agent=agent, state=saved_state, step=current_step, total_steps=total_steps)["action"]

        # calculate the numerator.
        numerators.append(max(action) - min(action))

        gradient_collector = []

        # generate noise samples in required norm based on the saved observation.
        noise_samples = noise_generator.get_samples(sample=saved_state, count=sample_number,
                                                    minimum_bounds=minimum_bounds, maximum_bounds=maximum_bounds)

        def calculate_handle(next_actions, former_actions, latter_actions):
            """
            Calculate gradient from a special strategy.

            :param next_actions: original next actions.
            :type next_actions: numpy.ndarray

            :param former_actions: positive perturbed actions (+h).
            :type former_actions: numpy.ndarray

            :param latter_actions: negative perturbed actions (-h).
            :type latter_actions: numpy.ndarray

            :return: gradient of special funciton.
            :rtype: numpy.ndarray
            """
            action_differences = former_actions - latter_actions
            return action_differences[argmax(next_actions)] - action_differences[argmin(next_actions)]

        for noise_sample in noise_samples:
            # obtain a perturbed estimated gradient based on the noise sample for further calculation.
            perturbed_gradient = task.estimate_gradient(agent=agent, state=noise_sample,
                                                        current_step=current_step, total_steps=total_steps,
                                                        calculate_handle=calculate_handle)
            gradient_collector.append(perturbed_gradient.flatten().tolist())

        denominators.append(gradient_collector)

        if verbose:
            monitor.output(current_step + 1, total_steps)

    scores, norm_value = [], {"L-1": inf, "L-2": 2, "L-inf": 1}[noise_generator.norm_type]
    for step, (numerator, denominator) in enumerate(zip(numerators, denominators)):
        if numerator > 0 and sum(denominator) > 0:
            # calculate norm values.
            estimated_location = max(linalg.norm(array(denominator), ord=norm_value, axis=1))
            scores.append(numerator / estimated_location)

    return scores
