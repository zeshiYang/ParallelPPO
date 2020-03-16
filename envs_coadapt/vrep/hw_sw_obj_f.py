import numpy as np
import time

import vrep_api.vrep as vrep
import traceback

from utils.vrep_exceptions import *
from cpg.cpg_gaits import DualTripod
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


class HwSwDistSim(DistanceSimulation):
    def scale_walker(self, x):
        self.walker.scale_vleg(leg_mapping["front_l"], x[0])
        self.walker.scale_vleg(leg_mapping["front_r"], x[0])
        self.walker.scale_vleg(leg_mapping["mid_l"], x[1])
        self.walker.scale_vleg(leg_mapping["mid_r"], x[1])
        self.walker.scale_vleg(leg_mapping["rear_l"], x[2])
        self.walker.scale_vleg(leg_mapping["rear_r"], x[2])

    def get_obj_f(self, max_steps, gait=DualTripod):

        def objective(x, cache_walker=True):
            """
            x[0] = cpg frequency [1, 45]
            x[1] = cpg phase offset [-np.pi, np.pi]
            x[2] = left bias of walker [0, 1]
            x[3] = right bias of walker [0, 1]
            x[4] = front pair leg scaling [.8, 1.2]
            x[5] = middle pair leg scaling [.8, 1.2]
            x[6] = rear pair leg scaling [.8, 1.2]
            :return:
            """
            self.start()
            x = np.asarray(x)[0]
            print('\nParameters: ' + str(x))
            try:
                # Clean up VREP
                if cache_walker and \
                        self.last_walker_params is not None and \
                        np.array_equal(x[4:], self.last_walker_params):
                    self.run()
                    self.load_prev_walker()
                else:
                    self.exit()
                    self.start()
                    self.load_scene()
                    self.load_walker()
                    self.run()

                    # Assign parameters
                    self.scale_walker(x[4:])

                    # Cache walker
                    self.walker_filename = "logs/models/{}.ttm" \
                        .format(time.strftime("%Y.%m.%d-%H.%M.%S"))
                    self.last_walker_params = x[4:]
                    self.save_walker(self.walker_filename)

                # Set the other parameters
                self.set_cpg(gait, x[0], x[1])
                self.walker.set_left_bias(x[2])
                self.walker.set_right_bias(x[3])

                # Get starting position
                start = self.get_pos(self.walker.base_handle)

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
