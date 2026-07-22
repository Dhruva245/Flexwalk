#!/usr/bin/env python
#
# *********     Gen Write Example (Multi-Motor)      *********
#
#
# Available ST Servo model on this example : All models using Protocol ST
# This example is tested with a ST Servo(ST3215/ST3020/ST3025), and an URT
# Controls 6 daisy-chained servos simultaneously in wheel (continuous rotation) mode
#

import sys
import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()

else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

sys.path.append("..")
from scservo_sdk import *                 # Uses SC Servo SDK library

# Default setting
SCS_IDS                     = [1, 2, 3, 4, 5, 6]  # List of all 6 servo IDs on the daisy chain
BAUDRATE                    = 1000000             # SC Servo default baudrate : 1000000
DEVICENAME                  = 'COM5'              # Check which port is being used on your controller
                                                   # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
SCS_MOVING_SPEED0           = 2400        # SC Servo moving speed (forward)
SCS_MOVING_SPEED1           = -2400       # SC Servo moving speed (reverse)
SCS_MOVING_ACC              = 0           # SC Servo moving acc -- 0 = instant stop/start, no ramping
                                           # (raise this later once basic on/off works, if you want smoother accel)

index = 0
scs_move_speed = [SCS_MOVING_SPEED0, 0, SCS_MOVING_SPEED1, 0]

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Get methods and members of Protocol
packetHandler = sms_sts(portHandler)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

def stop_all_motors():
    """Send speed 0 to every motor -- used on exit so nothing keeps spinning."""
    for scs_id in SCS_IDS:
        scs_comm_result, scs_error = packetHandler.WriteSpec(scs_id, 0, SCS_MOVING_ACC)
        if scs_comm_result != COMM_SUCCESS:
            print("[ID:%03d] %s" % (scs_id, packetHandler.getTxRxResult(scs_comm_result)))
        if scs_error != 0:
            print("[ID:%03d] %s" % (scs_id, packetHandler.getRxPacketError(scs_error)))

# Put every motor into wheel mode -- one call per motor ID
for scs_id in SCS_IDS:
    scs_comm_result, scs_error = packetHandler.WheelMode(scs_id)
    if scs_comm_result != COMM_SUCCESS:
        print("[ID:%03d] %s" % (scs_id, packetHandler.getTxRxResult(scs_comm_result)))
    elif scs_error != 0:
        print("[ID:%03d] %s" % (scs_id, packetHandler.getRxPacketError(scs_error)))

try:
    while 1:
        print("Press any key to continue! (or press ESC to quit!)")
        key = getch()
        if key == chr(0x1b):
            # ESC pressed -- make sure every motor is stopped before quitting
            print("Stopping all motors...")
            stop_all_motors()
            break

        # Write SC Servo goal speed/acc to every motor in the chain
        for scs_id in SCS_IDS:
            scs_comm_result, scs_error = packetHandler.WriteSpec(scs_id, scs_move_speed[index], SCS_MOVING_ACC)
            if scs_comm_result != COMM_SUCCESS:
                print("[ID:%03d] %s" % (scs_id, packetHandler.getTxRxResult(scs_comm_result)))
            if scs_error != 0:
                print("[ID:%03d] %s" % (scs_id, packetHandler.getRxPacketError(scs_error)))

        # Change move speed
        index += 1
        if index == 4:
            index = 0

except KeyboardInterrupt:
    # Safety net: if the script is killed with Ctrl+C, stop motors before exiting anyway
    print("Interrupted -- stopping all motors...")
    stop_all_motors()

# Close port
portHandler.closePort()
