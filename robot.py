import numpy as np
from dynamixel import Dynamixel, OperatingMode, ReadAttribute
import time
from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD
from enum import Enum, auto
from typing import Union


class MotorControlType(Enum):
    PWM = auto()
    POSITION_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()


class Robot:
    def __init__(self, device_name: str, baudrate=1_000_000, servo_ids=[0, 1, 2, 3, 4, 5]):
    # def __init__(self, dynamixel, baudrate=1_000_000, servo_ids=[1, 2, 3, 4, 5]):
        self.servo_ids = servo_ids
        # self.dynamixel = dynamixel
        self.dynamixel = Dynamixel.Config(baudrate=baudrate, device_name=device_name).instantiate()
        self.position_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.POSITION.value,
            4)
        for id in self.servo_ids:
            self.position_reader.addParam(id)

        self.velocity_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.VELOCITY.value,
            4)
        for id in self.servo_ids:
            self.velocity_reader.addParam(id)

        self.pos_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_POSITION,
            4)
        for id in self.servo_ids:
            self.pos_writer.addParam(id, [2048])

        self.current_limit_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            38,
            2)
        self.opmode_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            11,
            1)

        self.gain_p_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            84,
            2)
        self.gain_i_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            82,
            2)
        self.gain_d_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            80,
            2)
        for id in self.servo_ids:
            print(f'setting gain for {id}')

            if id < 2:
                p_data = [DXL_LOBYTE(640), DXL_HIBYTE(640)]
                i_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
                d_data = [DXL_LOBYTE(3600), DXL_HIBYTE(3600)]
            elif id == 2:
                p_data = [DXL_LOBYTE(1500), DXL_HIBYTE(1500)]
                i_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
                d_data = [DXL_LOBYTE(600), DXL_HIBYTE(600)]
            else:
                # Format the gains as 2-byte values
                p_data = [DXL_LOBYTE(400), DXL_HIBYTE(400)]
                i_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
                d_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]

                # Set current limit
                # self.current_limit_writer.addParam(id, [DXL_LOBYTE(-1250), DXL_HIBYTE(1250)])
                # self.current_limit_writer.txPacket()

                # Read back the current limit to verify
                # current_limit, dxl_comm_result, dxl_error = self.dynamixel.packetHandler.read2ByteTxRx(
                #     self.dynamixel.portHandler, id, 38)
                # print(f'current limit for {id} is {current_limit}')
            
            # Use addParam for first time, or changeParam if already added
            self.gain_p_writer.addParam(id, p_data)
            self.gain_i_writer.addParam(id, i_data)
            self.gain_d_writer.addParam(id, d_data)
            
            self.gain_p_writer.txPacket()
            self.gain_i_writer.txPacket()
            self.gain_d_writer.txPacket()
            
            # Verify the gains were set correctly
            p, i, d = self.read_gains(id)
            print(f'Verified gains for motor {id}: P={p}, I={i}, D={d}')

            if id == 4:
                self.opmode_writer.addParam(id, [DXL_LOBYTE(DXL_LOWORD(5))])
            else:
                self.opmode_writer.addParam(id, [DXL_LOBYTE(DXL_LOWORD(4))])
            self.opmode_writer.txPacket()

        self.pwm_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_PWM,
            2)
        for id in self.servo_ids:
            self.pwm_writer.addParam(id, [2048])
        self._disable_torque()
        self.motor_control_state = MotorControlType.DISABLED

    def read_position(self, tries=2):
        """
        Reads the joint positions of the robot. 2048 is the center position. 0 and 4096 are 180 degrees in each direction.
        :param tries: maximum number of tries to read the position
        :return: list of joint positions in range [0, 4096]
        """
        result = self.position_reader.txRxPacket()
        if result != 0:
            if tries > 0:
                return self.read_position(tries=tries - 1)
            else:
                print(f'failed to read position!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        positions = []
        for id in self.servo_ids:
            position = self.position_reader.getData(id, ReadAttribute.POSITION.value, 4)
            if position > 2 ** 31:
                position -= 2 ** 32
            positions.append(position)
        return positions

    # def read_velocity(self):
    #     """
    #     Reads the joint velocities of the robot.
    #     :return: list of joint velocities,
    #     """
    #     self.velocity_reader.txRxPacket()
    #     velocties = []
    #     for id in self.servo_ids:
    #         velocity = self.velocity_reader.getData(id, ReadAttribute.VELOCITY.value, 4)
    #         if velocity > 2 ** 31:
    #             velocity -= 2 ** 32
    #         velocties.append(velocity)
    #     return velocties

    def set_goal_pos(self, action):
        """

        :param action: list or numpy array of target joint positions in range [0, 4096]
        """
        if not self.motor_control_state is MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          DXL_LOBYTE(DXL_HIWORD(action[i])),
                          DXL_HIBYTE(DXL_HIWORD(action[i]))]
            self.pos_writer.changeParam(motor_id, data_write)

        self.pos_writer.txPacket()

    # def set_pwm(self, action):
    #     """
    #     Sets the pwm values for the servos.
    #     :param action: list or numpy array of pwm values in range [0, 885]
    #     """
    #     if not self.motor_control_state is MotorControlType.PWM:
    #         self._set_pwm_control()
    #     for i, motor_id in enumerate(self.servo_ids):
    #         data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
    #                       DXL_HIBYTE(DXL_LOWORD(action[i])),
    #                       ]
    #         self.pwm_writer.changeParam(motor_id, data_write)

    #     self.pwm_writer.txPacket()

    # def set_trigger_torque(self):
    #     """
    #     Sets a constant torque torque for the last servo in the chain. This is useful for the trigger of the leader arm
    #     """
    #     self.dynamixel._enable_torque(self.servo_ids[-1])
    #     self.dynamixel.set_pwm_value(self.servo_ids[-1], 200)

    # def limit_pwm(self, limit: Union[int, list, np.ndarray]):
    #     """
    #     Limits the pwm values for the servos in for position control
    #     @param limit: 0 ~ 885
    #     @return:
    #     """
    #     if isinstance(limit, int):
    #         limits = [limit, ] * 5
    #     else:
    #         limits = limit
    #     self._disable_torque()
    #     for motor_id, limit in zip(self.servo_ids, limits):
    #         self.dynamixel.set_pwm_limit(motor_id, limit)
    #     self._enable_torque()

    def _disable_torque(self):
        print(f'disabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def _enable_torque(self):
        print(f'enabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._enable_torque(motor_id)

    # def _set_pwm_control(self):
    #     self._disable_torque()
    #     for motor_id in self.servo_ids:
    #         self.dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
    #     self._enable_torque()
    #     self.motor_control_state = MotorControlType.PWM

    def _set_position_control(self):
        print('setting position control')
        self._disable_torque()
        for motor_id in self.servo_ids[:-1]:
            # self.dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
            if motor_id == 4:
                self.opmode_writer.changeParam(motor_id, [DXL_LOBYTE(DXL_LOWORD(5))])
            else:
                self.opmode_writer.changeParam(motor_id, [DXL_LOBYTE(DXL_LOWORD(4))])

            if motor_id < 2:
                p_data = [DXL_LOBYTE(640), DXL_HIBYTE(640)]
                i_data = [DXL_LOBYTE(200), DXL_HIBYTE(300)]
                d_data = [DXL_LOBYTE(3600), DXL_HIBYTE(3600)]
            # elif motor_id == 2:
            #     p_data = [DXL_LOBYTE(1500), DXL_HIBYTE(1500)]
            #     i_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
            #     d_data = [DXL_LOBYTE(600), DXL_HIBYTE(600)]
            elif motor_id < 5:
                p_data = [DXL_LOBYTE(640), DXL_HIBYTE(640)]
                i_data = [DXL_LOBYTE(300), DXL_HIBYTE(300)]
                d_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
            else:
                # Format the gains as 2-byte values
                p_data = [DXL_LOBYTE(400), DXL_HIBYTE(400)]
                i_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
                d_data = [DXL_LOBYTE(0), DXL_HIBYTE(0)]
            
            # Use addParam for first time, or changeParam if already added
            self.gain_p_writer.changeParam(motor_id, p_data)
            self.gain_i_writer.changeParam(motor_id, i_data)
            self.gain_d_writer.changeParam(motor_id, d_data)

        self.opmode_writer.txPacket()
        self.gain_p_writer.txPacket()
        self.gain_i_writer.txPacket()
        self.gain_d_writer.txPacket()

        self._enable_torque()
        self.motor_control_state = MotorControlType.POSITION_CONTROL

    def read_gains(self, motor_id):
        """
        Reads the current PID gains for a specific motor
        :param motor_id: ID of the motor to read gains from
        :return: tuple of (P gain, I gain, D gain)
        """
        p_gain, dxl_comm_result, dxl_error = self.dynamixel.packetHandler.read2ByteTxRx(
            self.dynamixel.portHandler, motor_id, 84)
        i_gain, dxl_comm_result, dxl_error = self.dynamixel.packetHandler.read2ByteTxRx(
            self.dynamixel.portHandler, motor_id, 82)
        d_gain, dxl_comm_result, dxl_error = self.dynamixel.packetHandler.read2ByteTxRx(
            self.dynamixel.portHandler, motor_id, 80)
        return p_gain, i_gain, d_gain


import pickle
import time

def record(robot, save=True):
    robot._disable_torque()

    try:
        actions = []
        while True:
            s = time.time()
            pos = robot.read_position()
            elapsed = time.time() - s
            actions.append(pos)
            print(f'read took {elapsed} pos {pos}')
            time.sleep(1/50)
    except KeyboardInterrupt:
        print('writing file')
        if save:
            pickle.dump(actions, open('actions.pkl', 'wb'))


def replay(robot):
    robot._enable_torque()

    actions = pickle.load(open('actions.pkl', 'rb'))
    robot.set_goal_pos(actions[0])
    time.sleep(3)

    while True:
        for action in actions:
            robot.set_goal_pos(action)
            time.sleep(1/50)

if __name__ == "__main__":
    robot = Robot(device_name='/dev/ttyACM0')
    import sys

    if sys.argv[1] == 'record':
        record(robot)
    elif sys.argv[1] == 'move':
        record(robot, save=False)
    else:
        robot._enable_torque()
        print('playing')
        time.sleep(3)
        replay(robot)
