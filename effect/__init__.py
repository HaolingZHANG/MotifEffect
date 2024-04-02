"""
@Author      : Haoling Zhang
@Description : Initialization of "effect" package
"""
from datetime import datetime


class Monitor(object):
    """
    Observe the progress of program operation.
    """

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def __call__(self,
                 moment: int,
                 finish: int,
                 extra: dict = None):
        """
        Output the current state of process.

        :param moment: current state of process.
        :type moment: int

        :param finish: total state of process.
        :type finish: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if self.last_time is None:
            self.last_time = datetime.now()

        if moment == 0:
            self.last_time = datetime.now()
            return

        position = int(moment / finish * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (finish - moment) / moment)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(finish)) - len(str(moment))) + str(moment) + "/" + str(finish)

        if moment < finish:
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

        if moment >= finish:
            self.last_time = None
            print()


from effect.networks import RestrictedWeight, RestrictedBias, NeuralMotif  # noqa
from effect.operations import prepare_data, prepare_motifs, calculate_landscape, calculate_gradients  # noqa
from effect.operations import generate_motifs, generate_outputs, calculate_differences  # noqa
from effect.robustness import estimate_lipschitz_by_motif, estimate_lipschitz, evaluate_propagation  # noqa
from effect.similarity import maximum_minimum_loss_search, minimum_loss_search  # noqa
from effect.similarity import execute_catch_processes, execute_escape_processes  # noqa
from effect.topography import find_ridge_line, find_valley_line  # noqa
