"""
@Author      : Haoling Zhang
@Description : Definition of task.
"""
from copy import deepcopy
from gym import make, Env
from itertools import product

from gym.envs.classic_control import CartPoleEnv
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from numpy import ndarray, array, linspace, argmax, abs, max, sum, mean, where
# noinspection PyPackageRequirements
from neat.config import Config
from sklearn.metrics import classification_report
from typing import Union, Tuple
from warnings import simplefilter

from practice.noise import NormNoiseGenerator
from practice.agent import NEATAgent, DefaultAgent

# ignore some warns in the step function of OpenAI gym environment.
simplefilter("ignore", UserWarning)
simplefilter("ignore", DeprecationWarning)


class BasicTask(object):

    def __init__(self,
                 description: str):
        """
        Initialize the Basic task.

        :param description: description of the task.
        :type description: str
        """
        self.description, self.noise_generator, self.experiences = description, None, []

    def set_noise(self,
                  noise_generator: NormNoiseGenerator):
        """
        Set the noise generator.

        :param noise_generator: generator of noise.
        :type noise_generator: practice.noise.NormNoiseGenerator
        """
        self.noise_generator = noise_generator

    def get_experiences(self) \
            -> list:
        return self.experiences

    def reset_experiences(self):
        self.experiences = []


class GymTask(BasicTask):

    def __init__(self,
                 environment: Env,
                 description: str,
                 maximum_generation: int,
                 iterations: int,
                 total_steps: int,
                 need_frames: bool = False):
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
        super(GymTask, self).__init__(description)
        self.environment, self.need_frames = environment, need_frames
        self.iterations, self.total_steps, self.maximum_generation = iterations, total_steps, maximum_generation
        self.action_handle, self.record_handle = None, None

    def set_action_handle(self, action_handle):
        """
        Set action handle (e.g. argmax)

        :param action_handle: action handle.
        """
        self.action_handle = action_handle

    def run(self,
            agent: DefaultAgent,
            random_seeds: Union[ndarray, list, None] = None) \
            -> dict:
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

    def run_1_iteration(self,
                        agent: DefaultAgent,
                        random_seed: Union[int, None] = None) \
            -> dict:
        """
        Run the task for one iteration.

        :param agent: available agent.
        :type agent: practice.agent.DefaultAgent

        :param random_seed: random seed for initializing the environment.
        :type random_seed: int or None

        :return: result set.
        :rtype: dict
        """
        states, actions, rewards, noises, frames = [], [], [], [], []
        state, _ = self.environment.reset(seed=random_seed)

        for one_step in range(self.total_steps):
            states.append(state)

            if self.need_frames:
                frames.append(self.environment.render())

            record = self.run_1_step(agent, state)

            actions.append(record["action"])
            rewards.append(record["reward"])
            noises.append(record["noise"])

            if record["done"]:
                break
            else:
                state = record["state"]

        self.environment.close()

        return {"states": array(states), "actions": array(actions), "rewards": array(rewards),
                "noises": array(noises), "frames": array(frames)}

    def run_1_step(self,
                   agent: DefaultAgent,
                   state: ndarray,
                   reset: bool = False) \
            -> dict:
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

        current_state, reward, done, _, _ = self.environment.step(action)

        return {"state": array(current_state), "action": array(action_values), "reward": reward,
                "noise": array(actual_state - state), "done": done}

    def sampling_states(self,
                        sampling: int) \
            -> ndarray:
        lower_bounds, upper_bounds = self.get_state_range()
        variable_set = [linspace(lower_bound, upper_bound, sampling + 2)[1: -1] for lower_bound, upper_bound
                        in zip(lower_bounds, upper_bounds)]
        return array(list(product(*variable_set)))

    @staticmethod
    def get_state_range():
        raise NotImplementedError("\"get_state_range\" interface needs to be implemented.")

    @staticmethod
    def save_frames_in_gif(frames: ndarray,
                           path: str,
                           dpi: int = 200,
                           interval: int = 1,
                           fps: int = 10):
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

        # noinspection PyTypeChecker
        gif = FuncAnimation(fig=pyplot.gcf(), func=animate, frames=len(frames), interval=interval)
        gif.save(path, writer='pillow', fps=fps)
        pyplot.close()


class NEATCartPoleTask(GymTask):

    def __init__(self,
                 maximum_generation: int,
                 need_frames: bool = False):
        """
        Initialize the CartPole task for NEAT algorithm.

        :param maximum_generation: maximum generation for fitness.
        :type maximum_generation: int

        :param need_frames: need to draw the frame in the environment.
        :type need_frames: bool
        """
        # noinspection PyCompatibility
        super().__init__(CartPoleEnv(render_mode="rgb_array"), "CartPole", maximum_generation, 100, 200, need_frames)
        self.set_action_handle(action_handle=argmax)

    def genomes_fitness(self,
                        genomes: dict,
                        neat_config: Config):
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

            # noinspection PyCompatibility
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
    def calculate_fitness(reward_collector: Union[ndarray, list]) \
            -> float:
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
    def get_state_range() \
            -> Tuple[ndarray, ndarray]:
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
    def check_bounds() \
            -> Tuple[float, float]:
        task, maximum_velocity, maximum_palstance = make("CartPole-v0").unwrapped, 0, 0

        for position in linspace(-0.05, 0.05, 20):
            for velocity in linspace(-0.05, 0.05, 20):
                for angle in linspace(-0.05, 0.05, 20):
                    for palstance in linspace(-0.05, 0.05, 20):
                        task.reset()
                        task.state, final_state = array([position, velocity, angle, palstance]), None
                        for _ in range(200):  # replay to find the bound.
                            final_state, reward, done, _, _ = task.step(1)
                            if done:
                                break

                        maximum_velocity = max(maximum_velocity, abs(final_state[1]))
                        maximum_palstance = max(maximum_velocity, abs(final_state[3]))

        return maximum_velocity, maximum_palstance


class SupervisionTask(BasicTask):

    def __init__(self,
                 trained_inputs: ndarray,
                 trained_outputs: ndarray,
                 tested_inputs: ndarray,
                 tested_outputs: ndarray,
                 description: str,
                 label_number: int,
                 data_types: list,
                 data_ranges: list,
                 maximum_generation: int):
        """
        Initialize the supervision classification task.

        :param trained_inputs: trained output data (or features).
        :type trained_inputs: numpy.ndarray

        :param trained_outputs: trained output data (or labels).
        :type trained_outputs: numpy.ndarray

        :param tested_inputs: trained output data (or features).
        :type tested_inputs: numpy.ndarray

        :param tested_outputs: trained output data (or labels).
        :type tested_outputs: numpy.ndarray

        :param description: description of the task.
        :type description: str

        :param label_number: number of labels in output data.
        :type label_number: int

        :param data_types: type of input data in each column ("binary", "categorical", or "continuous").
        :type data_types: list

        :param data_ranges: range of input data in each column.
        :type data_ranges: numpy.ndarray
        """
        super(SupervisionTask, self).__init__(description)
        if len(trained_inputs) != len(trained_outputs):
            raise ValueError("The trained inputs and trained outputs must have the same length.")
        if len(tested_inputs) != len(tested_outputs):
            raise ValueError("The trained inputs and trained outputs must have the same length.")

        self.trained_inputs, self.trained_outputs = trained_inputs, trained_outputs
        self.tested_inputs, self.tested_outputs = tested_inputs, tested_outputs
        self.data_types, self.data_ranges, self.label_number = data_types, data_ranges, label_number
        self.action_handle, self.record_handle, self.maximum_generation = argmax, None, maximum_generation

    def run(self,
            agent: DefaultAgent,
            train_flag: bool = False):
        if self.noise_generator is not None:
            available = where(array([data_type.lower() == "continuous" for data_type in self.data_types]) == True)[0]
            actual_inputs = []

            # In order to match the actual situation, we only intervene in continuous data types.
            if train_flag:
                actual_outputs = self.trained_outputs.copy()
                for values in self.trained_inputs:
                    new_values = values.copy()
                    # noinspection PyTypeChecker
                    changed_values = self.noise_generator.get_samples(sample=new_values[available], count=1,
                                                                      minimum_bounds=self.data_ranges[available, 0],
                                                                      maximum_bounds=self.data_ranges[available, 1])
                    new_values[available] = changed_values
                    actual_inputs.append(new_values.tolist())
            else:
                actual_outputs = self.tested_outputs.copy()
                for values in self.tested_inputs:
                    new_values = values.copy()
                    # noinspection PyTypeChecker
                    changed_values = self.noise_generator.get_samples(sample=new_values[available], count=1,
                                                                      minimum_bounds=self.data_ranges[available, 0],
                                                                      maximum_bounds=self.data_ranges[available, 1])
                    new_values[available] = changed_values
                    actual_inputs.append(new_values.tolist())

            actual_inputs = array(actual_inputs)
        else:
            if train_flag:
                actual_inputs, actual_outputs = self.trained_inputs.copy(), self.trained_outputs.copy()
            else:
                actual_inputs, actual_outputs = self.tested_inputs.copy(), self.tested_outputs.copy()

        raw_results = []
        for actual_input, expected_output in zip(actual_inputs, actual_outputs):
            obtained_output, _ = agent.work(actual_input)
            raw_results.append([expected_output, obtained_output])
        raw_results = array(raw_results).T

        report = classification_report(y_true=raw_results[0], y_pred=raw_results[1],
                                       digits=self.label_number, output_dict=True)
        # noinspection PyTypeChecker
        return {"fitness": report["weighted avg"]["f1-score"], "report": report, "raw": raw_results}

    def genomes_fitness(self,
                        genomes: dict,
                        neat_config: Config):
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

            # noinspection PyCompatibility
            collector = self.run(agent, train_flag=True)
            model_genome.fitness = collector["fitness"]
            situation.append(model_genome.fitness)

            if best_genome is None or model_genome.fitness > best_genome.fitness:
                best_genome = model_genome
                if self.record_handle is not None:
                    best_agent = self.record_handle(agent)
                else:
                    best_agent = deepcopy(agent)

            del agent, collector

        self.experiences.append((best_agent, situation))
