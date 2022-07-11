from gym import make
from itertools import product
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from numpy import ndenumerate, array, copy, zeros, linspace, argmax, abs, min, max, sum, mean
from warnings import simplefilter

from grace.agent import NEATAgent
from grace.handle import Monitor

# ignore some warns in the step function of OpenAI gym environment.
simplefilter("ignore", UserWarning)
simplefilter("ignore", DeprecationWarning)


class GymTask(object):

    def __init__(self, environment, description, maximum_generation, iterations, total_steps,
                 noise_generator=None, need_frames=False, verbose=True):
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

        :param noise_generator: generator of noise.
        :type noise_generator: grace.robust.NormNoiseGenerator

        :param verbose: need to show process log.
        :type verbose: bool
        """
        self.environment, self.description, self.noise_generator = environment, description, noise_generator
        self.iterations, self.total_steps, self.maximum_generation = iterations, total_steps, maximum_generation
        self.need_frames, self.verbose, self.monitor = need_frames, verbose, Monitor()
        self.experiences, self.record_handle = [], None

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
            if self.verbose:
                print("run iteration " + str(one_iteration + 1) + ":")

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
                if self.verbose:
                    self.monitor.output(one_step + 1, one_step + 1)
                break
            else:
                state = record["state"]

            if self.verbose:
                self.monitor.output(one_step + 1, self.total_steps)

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
            actual_state = self.noise_generator.get_samples(copy(state), 1, minimum_bounds, maximum_bounds)
        else:
            actual_state = copy(state)

        action, action_values = agent.work(actual_state)

        current_state, reward, done, _ = self.environment.step(action)

        return {"state": array(current_state), "action": array(action_values), "reward": reward,
                "noise": array(actual_state - state), "done": done}

    def estimate_gradient(self, agent, state, difference=1e-6, threshold=1e-6, calculate_handle=None):
        """
        Estimate gradient of the agent in the task.

        :param agent: available agent.
        :type agent: grace.DefaultAgent

        :param state: current state or observation.
        :type state: numpy.ndarray

        :param difference: minimum perturbation for estimating gradient.
        :type difference: float

        :param threshold: maximum difference between former and latter estimated gradients.
        :type threshold: float

        :param calculate_handle: extra calculate handle of gradient estimation, if required.
        :type calculate_handle: function

        :return: estimated gradient.
        :rtype: numpy.ndarray
        """
        gradient, perturbed_bounds = zeros(state.shape, dtype=state.dtype), []
        minimum_bounds, maximum_bounds = self.get_state_range()
        for state_value, lower_bound, upper_bound in zip(state, minimum_bounds, maximum_bounds):
            perturbed_bounds.append(min([upper_bound - state_value, state_value - lower_bound]))

        for position, perturbed_value in ndenumerate(perturbed_bounds):
            if perturbed_value == 0.0:
                continue

            gradient_record = []
            while True:
                perturbation = zeros(state.shape, dtype=state.dtype)
                perturbation[position] = perturbed_value

                d_actions = self.run_1_step(agent, state, True)["action"]
                p_actions = self.run_1_step(agent, state + perturbation, True)["action"]
                n_actions = self.run_1_step(agent, state - perturbation, True)["action"]

                if argmax(d_actions) == argmax(p_actions) and argmax(d_actions) == argmax(n_actions):
                    if calculate_handle is None:
                        saved_gradient = (p_actions[position] - n_actions[position]) / (2.0 * perturbed_value)
                    else:
                        saved_gradient = calculate_handle(p_actions, n_actions) / (2.0 * perturbed_value)

                    gradient_record.append(saved_gradient)

                if len(gradient_record) > 2 and abs(gradient_record[-2] - gradient_record[-1]) < difference:
                    break

                if perturbed_value <= threshold:
                    break

                perturbed_value /= 2.0

            gradient[position] = gradient_record[-1]

        return gradient

    def get_experience(self):
        return self.experiences

    def reset_experience(self, record_handle):
        self.experiences, self.record_handle = [], record_handle

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

    def __init__(self, maximum_generation, noise_generator=None, need_frames=False, verbose=True):
        """
        Initialize the CartPole task for NEAT algorithm.

        :param maximum_generation: maximum generation for fitness.
        :type maximum_generation: int

        :param noise_generator: generator of noise.
        :type noise_generator: grace.robust.NormNoiseGenerator or None

        :param need_frames: need to draw the frame in the environment.
        :type need_frames: bool

        :param verbose: need to show process log.
        :type verbose: bool
        """
        super().__init__(make("CartPole-v0").unwrapped, "CartPole", maximum_generation, 100, 200,
                         noise_generator, need_frames, verbose)

    def genomes_fitness(self, genomes, neat_config):
        """
        Calculate the fitness of the investigated genomes.

        :param genomes: NEAT genomes.

        :param neat_config: configure of NEAT algorithm.
        """
        best_genome, experience = None, None
        for genome_id, model_genome in genomes:
            agent = NEATAgent(model_genome, neat_config, "temp", action_handle=argmax)
            collector = super().run(agent)
            reward_collector = collector["rewards"]

            accumulative_rewards = []
            for rewards in reward_collector:
                accumulative_rewards.append(sum(rewards))

            model_genome.fitness = mean(accumulative_rewards)

            if best_genome is None or model_genome.fitness > best_genome.fitness:
                best_genome = model_genome
                experience = self.record_handle(agent)

            del agent, collector, reward_collector

        self.experiences.append(experience)

    @staticmethod
    def get_state_range():
        """
        Get the range of state.

        :return: the state range.
        :rtype: numpy.ndarray

        .. note::
            The bound of second and fourth values were tested (not infinite see below).
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
