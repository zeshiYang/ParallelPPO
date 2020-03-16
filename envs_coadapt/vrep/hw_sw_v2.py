import numpy as np
import time

import vrep_api.vrep as vrep
import traceback

from utils.vrep_exceptions import *
from cpg.cpg_gaits import Gait
from cpg.cpg import CpgController
from .normal_obj_f import DistanceSimulation

"""
old walker: front_l = 5, rear_l = 3
new walker: front_l = 3, rear_l = 5
"""
leg_mapping = {"front_l": 3,
               "front_r": 2,
               "mid_l": 4,
               "mid_r": 1,
               "rear_l": 5,
               "rear_r": 0}

phase_groupings = {
    0: 0,
    1: np.pi,
    2: 0,
    3: np.pi,
    4: 0,
    5: np.pi
}


def build_gait(x):
    """
    x[0] = f # Frequency
    x[1] = R_l # Sequence of leg extension length. scalar
    x[2] = R_f # Sequence offoot extension lengths. scalar
    x[3] = X_l # Sequence of leg offset values. scalar
    x[4] = X_f # Sequence of foot offset values. scalar
    x[5] = phase_offset # phase difference between leg and foot. scalar
    :param x:
    :return:
    """
    f = x[0]
    phase_offset = x[1]
    R_l = [x[2]] * 6 # todo here means the oscilator networks whill have 6 same legs and 6 same feet
    R_f = [x[3]] * 6
    X_l = [x[4]] * 6
    X_f = [x[5]] * 6

    gait = Gait(phase_groupings=phase_groupings, f=f, R_l=R_l,
                R_f=R_f, X_l=X_l, X_f=X_f, phase_offset=phase_offset)

    return gait


def is_1D(x):
    x = np.array(x)
    return len(x.shape) == 1


class JointObj2(DistanceSimulation):
    def set_cpg(self, gait, f=None, phase_offset=None):
        self.cpg = CpgController(gait)
        for _ in range(1000):
            self.cpg.update(plot=False)

    def scale_walker(self, x):
        self.walker.scale_vleg(leg_mapping["front_l"], x[0])
        self.walker.scale_vleg(leg_mapping["front_r"], x[0])
        self.walker.scale_vleg(leg_mapping["mid_l"], x[1])
        self.walker.scale_vleg(leg_mapping["mid_r"], x[1])
        self.walker.scale_vleg(leg_mapping["rear_l"], x[2])
        self.walker.scale_vleg(leg_mapping["rear_r"], x[2])

    def get_obj_f(self, max_steps, gait=None):

        def objective(x, cache_walker=True):
            """
            8 sw, 3 hw
            x[0] = f # Frequency [1, 45]
            x[1] = phase_offset # phase difference btwn leg and foot. [-pi, pi]
            x[2] = R_l # Sequence of leg extension length. [0, .04]
            x[3] = R_f # Sequence of foot extension lengths. [0, .04]
            x[4] = X_l # Sequence of leg offset values. [0, .04]
            x[5] = X_f # Sequence of foot offset values. [0, .04]
            x[6] = left bias of walker [.5, 1]
            x[7] = right bias of walker [.5, 1]
            x[8] = front pair leg scaling [.8, 1.2]
            x[9] = middle pair leg scaling [.8, 1.2]
            x[10] = rear pair leg scaling [.8, 1.2]
            walker params = x[-3:]
            :return:
            """
            if not is_1D(x):
                x = np.asarray(x)[0]
            else:
                x = np.asarray(x)

            self.start()

            gait_params = x[:5] # todo here should be gait_params = x[:5] since building a gait only need 6 parameters
            left_bias = x[6]
            right_bias = x[7]
            walker_params = x[8:]
            print("walker params", walker_params)

            print('\nParameters: ' + str(x))
            try:
                # Clean up VREP
                if cache_walker and \
                        self.last_walker_params is not None and \
                        np.array_equal(walker_params, self.last_walker_params):
                    self.run()
                    self.load_prev_walker()
                else:
                    self.exit()
                    self.start()
                    self.load_scene()
                    self.load_walker()
                    self.run()

                    # Assign parameters
                    self.scale_walker(walker_params)

                    # Cache walker
                    self.walker_filename = "logs/models/{}.ttm" \
                        .format(time.strftime("%Y.%m.%d-%H.%M.%S"))
                    self.last_walker_params = walker_params
                    self.save_walker(self.walker_filename)

                # Set the other parameters
                self.set_cpg(build_gait(gait_params))
                self.walker.set_left_bias(left_bias)
                self.walker.set_right_bias(right_bias)

                # Get starting position
                start = self.get_pos(self.walker.base_handle) #todo what is self.walker.base_handle

                # Run the simulation
                print('Running trial...')
                self.walk(max_steps)

                # Calculate how far the robot walked
                end = self.get_pos(self.walker.base_handle)
                dist = self.calc_dist(start, end)
                print("Distance traveled: {}".format(dist))

                # Clean up VREP
                self.remove_walker()
                self.close_scene()
                self.stop()
                self.exit()

                return np.array([dist])

            except SceneLoadingError as e:
                print(str(e))
                self.stop()
                self.close_scene()

            except (ConnectionError,
                    WalkerLoadingError,
                    MotorLoadingError,
                    HandleLoadingError) as e:
                print("Encountered an exception: {} "
                      "disconnecting from remote API server"
                      .format(e))
                vrep.simxFinish(self.client_id)
                traceback.print_exc()
                self.stop()
                self.close_scene()
                raise e

        return objective
