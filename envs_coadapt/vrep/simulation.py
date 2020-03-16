import vrep_api.vrep as vrep

from utils.vrep_helpers import *
from utils.vrep_exceptions import *
from walkers.scalable_walker import ScalableWalker
from utils import DEFAULT_SCENE, DEFAULT_WALKER

VALID_ERROR_CODES = (0, 1)

OP_MODE = vrep.simx_opmode_blocking
#OP_MODE = vrep.simx_opmode_oneshot

LEG_NAME_TEMPLATES = ['HorizLeg9', 'VertLeg9', 'HorizLeg10', 'VertLeg10',
                      'HorizLeg11', 'VertLeg11', \
                      'HorizLeg12', 'VertLeg12', 'HorizLeg13', 'VertLeg13',
                      'HorizLeg14', 'VertLeg14']


class Simulation(object):
    def __init__(self, scene=DEFAULT_SCENE, walker_cl=DEFAULT_WALKER):
        self.client_id = -1
        self.scene = scene
        self.walker_cl = walker_cl
        self.walker = None
        self.walker_filename = None
        self.last_walker_params = None

    def start(self):
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.client_id == 1:
            raise ConnectionError
        vrep.simxSynchronous(self.client_id, 1)
        print('Initialized simulation environment with client id {}'.format(self.client_id))

    def load_scene(self, scene=None):
        if scene is None:
            scene = self.scene
        res = vrep.simxLoadScene(self.client_id, scene, 1, OP_MODE)
        if res not in VALID_ERROR_CODES:
            raise SceneLoadingError(scene, res)

        print("Loading scene: {}".format(self.scene))

    def load_walker(self, filename=None, state=0):
        """
        Load a saved model/walker
        #todo What is the difference between a walker in client side and a walker in server side?
        :param filename: filename
        :param state: 1 if file is on client side
        (if we want to load a walker not saved in simulation)
        0 if file is on server side
        (when we save a scaled walker)
        :return:
        """
        if filename is not None:
            print('Loading Walker: ' + filename + '...')
            err, _ = vrep.simxLoadModel(self.client_id, filename, state,
                                        OP_MODE)
            if err not in VALID_ERROR_CODES:
                raise WalkerLoadingError()
        self.walker = self.walker_cl()

    def load_prev_walker(self):
        self.load_walker(self.walker_filename, 0)

    def save_walker(self, name):
        """Saves to the server side"""
        print("Saving Walker {}".format(name))
        err, _, _, _, _ = \
            vrep.simxCallScriptFunction(self.client_id, "Walker",
                                        vrep.sim_scripttype_childscript,
                                        "saveModel_function",
                                        [self.walker.base_handle],
                                        [],
                                        [name],
                                        bytearray(), OP_MODE)
        if err not in VALID_ERROR_CODES:
            print(err)
            raise WalkerSaveError(name)
        print("Walker Saved")

    def remove_walker(self):
        print("Removing Walker...")
        err = vrep.simxRemoveModel(self.client_id, self.walker.base_handle,
                                   OP_MODE)
        if err not in VALID_ERROR_CODES:
            raise RemoveWalkerError()
        print("Walker Removed")

    def run(self):
        print("Running simulation")
        vrep.simxStartSimulation(self.client_id, OP_MODE)

    def pause(self):
        print("Pausing simulation")
        vrep.simxPauseSimulation(self.client_id, OP_MODE)

    def stop(self):
        print("Stopping simulation")
        vrep.simxStopSimulation(self.client_id, OP_MODE)
        vrep.simxGetPingTime(self.client_id)
        vrep.simxClearIntegerSignal(self.client_id, "", OP_MODE)
        vrep.simxClearStringSignal(self.client_id, "", OP_MODE)
        vrep.simxClearFloatSignal(self.client_id, "", OP_MODE)
        print("Exited simulation environment")

    def close_scene(self):
        print("Closing scene")
        vrep.simxCloseScene(self.client_id, OP_MODE)
        print('Closed Scene')

    def exit(self):
        print("Exiting VREP")
        vrep.simxFinish(self.client_id)

    def wait(self, steps):
        for _ in range(steps):
            vrep.simxSynchronousTrigger(self.client_id)

    def get_pos(self, handle):
        res, pos = vrep.simxGetObjectPosition(self.client_id, handle, -1,
                                              OP_MODE)
        if res not in VALID_ERROR_CODES:
            raise HandleLoadingError(handle)

        return pos


if __name__ == "__main__":
    sim = Simulation()
    sim.start()
    sim.load_scene()
    sim.load_walker()
    sim.run()
    sim.wait(10)
    sim.close_scene()
    sim.stop()
    sim.exit()
