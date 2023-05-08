from copy import deepcopy
from gym import make
from itertools import product
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from numpy import array, arange, linspace, random, repeat, expand_dims, argmax, abs, min, max, sum, mean, clip
from warnings import simplefilter

from practice.agent import NEATAgent

# ignore some warns in the step function of OpenAI gym environment.
simplefilter("ignore", UserWarning)
simplefilter("ignore", DeprecationWarning)


class GymTask(object):

    def __init__(self, environment, description, maximum_generation, iterations, total_steps, need_frames=False):
        """
        Initialize the OpenAI gym task.

        :param environment: Open-AI Gym task.
        :type environment: gym.Env

        :param description: description of the task.
        :type description: str

        :param maximum_generation: maximum generation for fitness.
        :type maximum_generation: int

        :param iterations: running iterations (for fitness).
        :type iterations: int

        :param total_steps: running total steps (for fitness) per iteration.
        :type total_steps: int
        """
        self.environment, self.description, self.noise_generator = environment, description, None
        self.iterations, self.total_steps, self.maximum_generation = iterations, total_steps, maximum_generation
        self.action_handle, self.need_frames, self.experiences, self.record_handle = None, need_frames, [], None

    def set_noise(self, noise_generator):
        """
        Set the noise generator.

        :param noise_generator: generator of noise.
        :type noise_generator: grace.robust.NormNoiseGenerator
        """
        self.noise_generator = noise_generator

    def set_action_handle(self, action_handle):
        """
        Set action handle (e.g. argmax)

        :param action_handle: action handle.
        """
        self.action_handle = action_handle

    def run(self, agent, random_seeds=None):
        """
        Run the task.

        :param agent: available agent.
        :type agent: grace.agent.DefaultAgent

        :param random_seeds: random seeds for initialize the environment.
        :type random_seeds: numpy.ndarray, list, or None

        :return: result set.
        :rtype: dict
        """
        state_collector, action_collector, reward_collector, frame_collector, noise_collector = [], [], [], [], []

        if random_seeds is not None:
            assert self.iterations == len(random_seeds)

        for one_iteration in range(self.iterations):
            if random_seeds is not None:
                record = self.run_1_iteration(agent, random_seeds[one_iteration])
            else:
                record = self.run_1_iteration(agent)

            state_collector.append(record["states"])
            action_collector.append(record["actions"])
            reward_collector.append(record["rewards"])
            noise_collector.append(record["noises"])
            if self.need_frames:
                frame_collector.append(record["frames"])

        return {"states": array(state_collector, dtype=object), "actions": array(action_collector, dtype=object),
                "rewards": array(reward_collector, dtype=object), "noises": array(noise_collector, dtype=object),
                "frames": array(frame_collector, dtype=object)}

    def run_1_iteration(self, agent, random_seed=None):
        """
        Run the task for one iteration.

        :param agent: available agent.
        :type agent: grace.agent.DefaultAgent

        :param random_seed: random seed for initializing the environment.
        :type random_seed: int or None

        :return: result set.
        :rtype: dict
        """
        self.environment.seed(random_seed)

        states, actions, rewards, noises, frames = [], [], [], [], []
        state = self.environment.reset()

        for one_step in range(self.total_steps):
            states.append(state)

            if self.need_frames:
                frames.append(self.environment.render(mode="rgb_array"))

            record = self.run_1_step(agent, state)

            actions.append(record["action"])
            rewards.append(record["reward"])
            noises.append(record["noise"])

            if record["done"]:
                break
            else:
                state = record["state"]

        self.environment.close()
        self.environment.seed(None)

        return {"states": array(states), "actions": array(actions), "rewards": array(rewards),
                "noises": array(noises), "frames": array(frames)}

    def run_1_step(self, agent, state, reset=False):
        """
        Run the task in one step.

        :param agent: available agent.
        :type agent: grace.agent.DefaultAgent

        :param state: state in task.
        :type state: numpy.ndarray

        :param reset: need to initialize or reset the task.
        :type reset: bool

        :return: result set.
        :rtype: dict
        """
        if reset:
            self.environment.reset()
            self.environment.state = state

        if self.noise_generator is not None:
            minimum_bounds, maximum_bounds = self.get_state_range()
            actual_state = self.noise_generator.get_samples(state.copy(), 1, minimum_bounds, maximum_bounds)
        else:
            actual_state = state.copy()

        action, action_values = agent.work(actual_state)

        current_state, reward, done, _ = self.environment.step(action)

        return {"state": array(current_state), "action": array(action_values), "reward": reward,
                "noise": array(actual_state - state), "done": done}

    def get_experiences(self):
        return self.experiences

    def reset_experiences(self):
        self.experiences = []

    def sampling_states(self, sampling):
        lower_bounds, upper_bounds = self.get_state_range()
        variable_set = [linspace(lower_bound, upper_bound, sampling + 2)[1: -1] for lower_bound, upper_bound
                        in zip(lower_bounds, upper_bounds)]
        return array(list(product(*variable_set)))

    @staticmethod
    def get_state_range():
        raise NotImplementedError("\"get_state_range\" interface needs to be implemented.")

    @staticmethod
    def save_frames_in_gif(frames, path, dpi=200, interval=1, fps=10):
        """
        Save frames in GIF file.

        :param frames: saved frames in the environment.
        :type frames: numpy.ndarray

        :param path: path to save file.
        :type path: str

        :param dpi: dots (pixels) per inch.
        :type dpi: int

        :param interval: interval between frames.
        :type interval: int

        :param fps: frames per second.
        :type fps: int
        """
        pyplot.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
        annotate = pyplot.annotate("step = 1", (1, 1))
        patch = pyplot.imshow(frames[0])
        pyplot.axis("off")

        def animate(index):
            patch.set_data(frames[index])
            annotate.set_text("step = " + str(index + 1))
            return patch

        gif = FuncAnimation(pyplot.gcf(), animate, frames=len(frames), interval=interval)
        gif.save(path, writer='pillow', fps=fps)
        pyplot.close()


class NEATCartPoleTask(GymTask):

    def __init__(self, maximum_generation, need_frames=False):
        """
        Initialize the CartPole task for NEAT algorithm.

        :param maximum_generation: maximum generation for fitness.
        :type maximum_generation: int

        :param need_frames: need to draw the frame in the environment.
        :type need_frames: bool
        """
        super().__init__(make("CartPole-v0").unwrapped, "CartPole", maximum_generation, 100, 200, need_frames)
        self.set_action_handle(action_handle=argmax)

    def genomes_fitness(self, genomes, neat_config):
        """
        Calculate the fitness of the investigated genomes.

        :param genomes: NEAT genomes.
        :type genomes: dict

        :param neat_config: configure of NEAT algorithm.
        :type neat_config: neat.config.Config
        """
        best_genome, best_agent, situation = None, None, []
        for genome_id, model_genome in genomes:
            agent = NEATAgent(model_genome, neat_config, "temp", action_handle=self.action_handle)

            collector = super().run(agent)
            model_genome.fitness = self.calculate_fitness(collector["rewards"])
            situation.append(model_genome.fitness)

            if best_genome is None or model_genome.fitness > best_genome.fitness:
                best_genome = model_genome
                if self.record_handle is not None:
                    best_agent = self.record_handle(agent)
                else:
                    best_agent = deepcopy(agent)

            del agent, collector

        self.experiences.append((best_agent, situation))

    @staticmethod
    def calculate_fitness(reward_collector):
        """
        Calculate the agent fitness from the reward collector.

        :param reward_collector: reward collector.
        :type reward_collector: numpy.ndarray or list

        :return: fitness.
        :rtype: float
        """
        accumulative_rewards = []
        for rewards in reward_collector:
            accumulative_rewards.append(sum(rewards))
        return mean(accumulative_rewards)

    @staticmethod
    def get_state_range():
        """
        Get the range of state.

        :return: the state range.
        :rtype: numpy.ndarray

        .. note::
            The bound of second and fourth intersected_values were tested (not infinite see below).
        """
        lower_bound = -array([4.8, 2.197613184073254, 0.418, 3.3345776572433])
        upper_bound = +array([4.8, 2.197613184073254, 0.418, 3.3345776572433])
        return lower_bound, upper_bound

    @staticmethod
    def check_bounds():
        task, maximum_velocity, maximum_palstance = make("CartPole-v0").unwrapped, 0, 0

        for position in linspace(-0.05, 0.05, 20):
            for velocity in linspace(-0.05, 0.05, 20):
                for angle in linspace(-0.05, 0.05, 20):
                    for palstance in linspace(-0.05, 0.05, 20):
                        task.reset()
                        task.state, final_state = array([position, velocity, angle, palstance]), None
                        for _ in range(200):  # replay to find the bound.
                            final_state, reward, done, _ = task.step(1)
                            if done:
                                break

                        maximum_velocity = max(maximum_velocity, abs(final_state[1]))
                        maximum_palstance = max(maximum_velocity, abs(final_state[3]))

        return maximum_velocity, maximum_palstance


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
