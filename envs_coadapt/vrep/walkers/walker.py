import vrep_api.vrep as vrep

CLIENTID = 0
VALID_ERROR_CODES = (0, 1)
GLOBALID = -1
MOTOR_NAME_TEMPLATES = ['flv', 'mlv', 'blv', 'brv', 'mrv', 'frv', 'flh', 'mlh', \
                        'blh', 'brh', 'mrh', 'frh']
DEFAULT_BIASES = (.925, 1)  # To correct for model imbalances


class Walker(object):
    def __init__(self, client=0, name='Walker'):
        assert client != -1, 'Invalid client ID'
        print('Loading walker...')
        CLIENTID = client
        global GLOBALID
        if GLOBALID != -1:
            self.id = str(GLOBALID)
            name += self.id
        else:
            self.id = ''

        errorCode, self.base_handle = vrep.simxGetObjectHandle(CLIENTID, name,
                                                               vrep.simx_opmode_blocking)
        errorCode, self.handle = vrep.simxGetCollectionHandle(CLIENTID, name,
                                                              vrep.simx_opmode_blocking)
        errorCode, self.motor_handle = vrep.simxGetCollectionHandle(CLIENTID,
                                                                    'Motors',
                                                                    vrep.simx_opmode_blocking)
        group_data = vrep.simxGetObjectGroupData(CLIENTID, self.handle, 0,
                                                 vrep.simx_opmode_blocking)
        assert group_data[0] in VALID_ERROR_CODES, 'Error loading walker'

        self.motor_names = [0 for _ in range(12)]
        self.handles = [0 for _ in range(12)]
        self.legs = []
        self.leg_names = [0 for _ in range(12)]
        self.leg_handles = [0 for _ in range(12)]
        self.horiz_names = [0 for _ in range(12)]
        self.horiz_handles = [0 for _ in range(12)]
        self.loading_legs(group_data)

        self.set_left_bias(DEFAULT_BIASES[0])
        self.set_right_bias(DEFAULT_BIASES[1])

    def loading_legs(self, group_data):
        for i in range(12):
            for j in range(len(group_data[4])):
                if group_data[4][j] == (MOTOR_NAME_TEMPLATES[i] + self.id):
                    self.handles[i] = group_data[1][j]
                    break
        for i in range(6):
            self.legs.append(
                Leg(Motor(self.handles[i]), Motor(self.handles[i + 6])))

    def set_left_bias(self, bias):
        for i in range(3):
            self.legs[i].bias_factor = bias

    def set_right_bias(self, bias):
        for i in range(3, 6):
            self.legs[i].bias_factor = bias

    def update_energy(self):
        group_data = vrep.simxGetObjectGroupData(CLIENTID, self.motor_handle,
                                                 15, vrep.simx_opmode_streaming)
        force_data = [0 for _ in range(12)]
        for i in range(len(group_data[1])):
            for j in range(len(self.handles)):
                if group_data[1][i] == self.handles[j]:
                    force_data[j] = group_data[3][(2 * i) + 1]

        for i in range(len(self.legs)):
            self.legs[i].vertical_motor.update_force(force_data[i])
            self.legs[i].horizontal_motor.update_force(force_data[i + 6])

    def calculate_energy(self):
        total_energy = 0
        for l in self.legs:
            total_energy += l.horizontal_motor.calculate_energy() \
                            + l.vertical_motor.calculate_energy()
        return total_energy

    def reset(self):
        for l in self.legs:
            l.reset()


class Leg(object):
    def __init__(self, vertical_motor, horizontal_motor):
        self.vertical_motor = vertical_motor
        self.horizontal_motor = horizontal_motor
        self.bias_factor = 1

    def extendX(self, dist, force_data=True):
        return self.horizontal_motor.setPos(-dist * self.bias_factor,
                                            force_data) # just set the next position, but not actually conducted by the engine.

    def extendZ(self, dist, force_data=True):
        return self.vertical_motor.setPos(-dist * self.bias_factor, force_data)

    def reset(self):
        self.horizontal_motor.reset()
        self.vertical_motor.reset()


class Motor:
    def __init__(self, handle):
        self.handle = handle
        self.total_distance = 0
        self.prev_pos = None
        self.total_force = 0
        self.samples = 0

    def getPos(self):
        errorCode, pos = vrep.simxGetJointPosition(CLIENTID, self.handle,
                                                   vrep.simx_opmode_streaming)
        assert errorCode in VALID_ERROR_CODES, 'Error getting motor position'
        return pos

    def setPos(self, pos, force_data=False):
        errorCode = vrep.simxSetJointTargetPosition(CLIENTID, self.handle, pos,
                                                    vrep.simx_opmode_streaming)
        assert errorCode in VALID_ERROR_CODES, 'Error setting motor position'

    def update_force(self, force):
        curr_pos = self.getPos()
        if self.prev_pos and curr_pos <= self.prev_pos:
            self.total_distance += self.prev_pos - curr_pos
        self.prev_pos = curr_pos
        if force > 0:
            self.total_force += force
            self.samples += 1

    def calculate_energy(self):
        if self.samples == 0:
            return 0
        else:
            return self.total_distance * self.total_force / self.samples

    def reset(self):
        self.total_distance = 0
        self.prev_pos = None
        self.total_force = 0
        self.samples = 0
