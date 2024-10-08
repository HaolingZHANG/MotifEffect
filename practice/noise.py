"""
@Author      : Haoling Zhang
@Description : Definition of noise scale.
"""
from numpy import ndarray, arange, random, repeat, expand_dims, min, max, clip


class NormNoiseGenerator(object):

    def __init__(self,
                 norm_type: str,
                 noise_level: float = 1.0,
                 noise_scale: float = 1.0):
        """
        Initialize the noise iterator based on norm.

        :param norm_type: perturbed norm type in the noise iterator.
        :type norm_type: str

        :param noise_level: proportion of samples with noise in all samples.
        :type noise_level: float

        :param noise_scale: degree of noise attenuation.
        :type noise_scale: float
        """
        self.norm_type, self.noise_level, self.noise_scale = norm_type, noise_level, noise_scale

    # noinspection PyArgumentList
    def get_samples(self,
                    sample: ndarray,
                    count: int,
                    minimum_bounds: ndarray,
                    maximum_bounds: ndarray) \
            -> ndarray:
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
            # noinspection PyArgumentEqualDefault
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
