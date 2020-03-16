from .walker import *
import math

CLIENTID = 0
"""
NEW WALKER
"""
LEG_NAME_TEMPLATES = ['VertLeg9', 'VertLeg10', 'VertLeg11', 'VertLeg12',
                      'VertLeg13', 'VertLeg14', \
                      'HorizLeg9', 'HorizLeg10', 'HorizLeg11', 'HorizLeg12',
                      'HorizLeg13', 'HorizLeg14']

SIDE_TEMPLATES = ['SideExt9', 'SideExt10', 'SideExt11', 'SideExt12',
                  'SideExt13', 'SideExt14', ' SideJoint9', 'SideJoint10',
                  'SideJoint11', 'SideJoint12',
                  'SideJoint13', 'SideJoint14']
"""
OLD WALKER
ker
LEG_NAME_TEMPLATES = ['VertLeg1', 'VertLeg2', 'VertLeg3', 'VertLeg6',
                      'VertLeg7', 'VertLeg8', \
                      'HorizLeg1', 'HorizLeg2', 'HorizLeg3', 'HorizLeg6',
                      'HorizLeg7', 'HorizLeg8']
"""
SIM_MODE = vrep.simx_opmode_blocking
POSITION_MODE = vrep.simx_opmode_oneshot
VALID_ERROR_CODES = (0, 1)


class ScalableWalker(Walker):
    def __init__(self):
        super(ScalableWalker, self).__init__()
        errorCode, self.position = \
            vrep.simxGetObjectPosition(CLIENTID,
                                       self.base_handle,
                                       -1,
                                       SIM_MODE)  # (2)

        print('Walker loaded')

    def loading_legs(self, group_data):
        for i in range(12):
            for j in range(len(group_data[4])):
                if group_data[4][j] == (MOTOR_NAME_TEMPLATES[i] + self.id):
                    self.motor_names[i] = group_data[4][j]
                    self.handles[i] = group_data[1][j]
                if group_data[4][j] == (LEG_NAME_TEMPLATES[i]):
                    self.leg_names[i] = group_data[4][j]
                    self.leg_handles[i] = group_data[1][j]
                if group_data[4][j] == (SIDE_TEMPLATES[i]):
                    self.horiz_names[i] = group_data[4][j]
                    self.horiz_handles[i] = group_data[1][j]
        for i in range(6):
            self.legs.append(LegGeometry(Motor(self.handles[i]),
                                         Motor(self.handles[i + 6]),
                                         self.leg_names[i],
                                         self.leg_names[i + 6],
                                         self.leg_handles[i],
                                         self.leg_handles[i + 6],
                                         self.base_handle, self.horiz_names[i],
                                         self.horiz_handles[i],
                                         self.horiz_names[i + 6],
                                         self.horiz_handles[i + 6],
                                         self.motor_names[i],
                                         self.motor_names[i + 6]))

    def scale_vleg(self, leg, scale):
        """

        :param leg: index of legs to scale.
            front left: 3
            front right: 2
            mid left: 4
            mid right: 1
            rear left: 5
            rear right: 0
        :param scale:
        :return:
        """
        assert .7 <= scale <= 1.4, "Scaling vertical too much! {}".format(scale)
        self.legs[leg].scale_vertical_leg(1, 1, scale)

    def scale_hleg(self, leg, scale):
        self.legs[leg].scale_horizontal_leg(1, 1, scale)


class LegGeometry(Leg):
    def __init__(self, vertical_motor, horizontal_motor, vertical_leg,
                 horizontal_leg,
                 vertical_handle, horizontal_handle, walker_handle, horiz_side,
                 horiz_side_handle,
                 horiz_joint, horiz_joint_handle, vmotorname, hmotorname):

        super(LegGeometry, self).__init__(vertical_motor, horizontal_motor)
        self.vertical_leg = vertical_leg
        self.horizontal_leg = horizontal_leg
        self.vertical_handle = vertical_handle
        self.horizontal_handle = horizontal_handle
        self.walker_handle = walker_handle
        self.horiz_side = horiz_side
        self.horiz_side_handle = horiz_side_handle
        self.horiz_joint = horiz_joint
        self.horiz_joint_handle = horiz_joint_handle
        self.vmotorname = vmotorname
        self.hmotorname = hmotorname

    def scale_vertical_leg(self, x, y, z):
        scale = [x, y, z]
        handles = self.ungroup(name=self.vertical_leg,
                               handle=self.vertical_handle)
        leg_handle = handles[0]
        position = self.get_position(leg_handle)
        diff = self.scale_leg(name=self.vertical_leg, scale=[x, y, z],
                              handle=leg_handle)
        self.reposition(leg_handle, diff[0], diff[1], scale, position, -1)
        self.group(name=self.vertical_leg, handles=handles)

        return diff

    def call_set_position(self, name, handles, position):
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(
            CLIENTID, name,
            vrep.sim_scripttype_childscript,
            'setObjectPosition_function',
            handles, position, [],
            bytearray(), SIM_MODE)

    def displacement(self, orientation, hyp):
        angle = math.radians(90) - orientation[1]
        disz = math.sin(angle) * hyp
        disx = math.cos(angle) * hyp
        return [disx, disz]

    def reposition(self, handle, diff_x, diff_z, scale, og_position, relative):
        if scale[2] > 1:
            new_position = [og_position[0] - diff_x, og_position[1],
                            og_position[2] - diff_z]
        else:
            new_position = [og_position[0] + diff_x, og_position[1],
                            og_position[2] + diff_z]
        vrep.simxSetObjectPosition(CLIENTID, handle, relative,
                                   new_position, POSITION_MODE)

    def get_handle(self, name):
        err, handle = vrep.simxGetObjectHandle(CLIENTID, name, SIM_MODE)
        return handle

    def get_position(self, handle):
        err, position = vrep.simxGetObjectPosition(CLIENTID,
                                                   handle,
                                                   -1,
                                                   SIM_MODE)
        return position

    def get_orientation(self, handle):
        err, orientation = vrep.simxGetObjectOrientation(CLIENTID, handle, -1,
                                                         SIM_MODE)
        return orientation

    def scale_leg(self, name, scale, handle):
        leg_handle = handle

        err, og_max_z = \
            vrep.simxGetObjectFloatParameter(CLIENTID,
                                             leg_handle,
                                             vrep.sim_objfloatparam_objbbox_max_z,
                                             SIM_MODE)

        err, og_max_x = \
            vrep.simxGetObjectFloatParameter(CLIENTID,
                                             leg_handle,
                                             vrep.sim_objfloatparam_objbbox_max_x,
                                             SIM_MODE)

        res, retInts, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(CLIENTID, name,
                                        vrep.sim_scripttype_childscript,
                                        'scaleObject_function',
                                        [leg_handle], scale, [],
                                        bytearray(), SIM_MODE)

        if res != vrep.simx_return_ok:
            raise Exception(
                'Remote function (scale_leg) call failed, handle: ' + str(
                    retInts))

        err, new_max_z = \
            vrep.simxGetObjectFloatParameter(CLIENTID,
                                             leg_handle,
                                             vrep.sim_objfloatparam_objbbox_max_z,
                                             SIM_MODE)

        err, new_max_x = \
            vrep.simxGetObjectFloatParameter(CLIENTID,
                                             leg_handle,
                                             vrep.sim_objfloatparam_objbbox_max_x,
                                             SIM_MODE)

        diff_x = abs(new_max_x - og_max_x)
        diff_z = abs(new_max_z - og_max_z)

        return [diff_x, diff_z]

    def ungroup(self, name, handle):
        res, handles, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(CLIENTID, name,
                                        vrep.sim_scripttype_childscript,
                                        'ungroupShape_function',
                                        [handle], [], [],
                                        bytearray(), SIM_MODE)
        if res != vrep.simx_return_ok:
            raise Exception(
                'Remote function (ungroup_shape) call failed, handle: ' + str(
                    handle))

        return handles

    def group(self, name, handles):
        res, handle, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(CLIENTID, name,
                                        vrep.sim_scripttype_childscript,
                                        'groupShapes_function',
                                        handles, [], [],
                                        bytearray(), SIM_MODE)
        if res != vrep.simx_return_ok:
            raise Exception(
                'Remote function (group_shapes) call failed, handle: ' + str(
                    handles))
