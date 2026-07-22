#!/usr/bin/env python
#
# *********  360 Degree Rotation Per Keypress - All 6 Motors  *********
#
# Each time you press a key, all 6 motors rotate exactly 360 degrees
# (alternating direction) and then stop, waiting for the next keypress.
# Press ESC to quit.
#

import sys
import os
import time

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
from scservo_sdk import *                      # Uses SC Servo SDK library

# Default setting
SCS_IDS                     = [1, 2, 3, 4, 5, 6]  # Your 6 servo IDs
BAUDRATE                    = 1000000             # SC Servo default baudrate : 1000000
DEVICENAME                  = 'COM5'              # Check which port is being used on your controller

SCS_MOVING_SPEED            = 2400              # SC Servo moving speed
SCS_MOVING_ACC              = 50                # SC Servo moving acc

MODE_REGISTER                = 33               # Feetech STS/SMS "Mode" control table address
POSITION_MODE_VALUE          = 0                # 0 = position mode, 1 = wheel mode

# Two endpoints -- one full 360 degree turn between them, alternating direction each press
scs_goal_position = [0, 4095]
index = 0

# Initialize PortHandler / PacketHandler
portHandler = PortHandler(DEVICENAME)
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

# --- SAFETY STEP: force every motor OUT of wheel mode back into position mode ---
print("Setting all motors to position mode...")
for scs_id in SCS_IDS:
    scs_comm_result, scs_error = packetHandler.write1ByteTxRx(scs_id, MODE_REGISTER, POSITION_MODE_VALUE)
    if scs_comm_result != COMM_SUCCESS:
        print("[ID:%03d] %s" % (scs_id, packetHandler.getTxRxResult(scs_comm_result)))
    elif scs_error != 0:
        print("[ID:%03d] %s" % (scs_id, packetHandler.getRxPacketError(scs_error)))

while 1:
    print("Press any key to rotate 360 degrees once! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    # Add every motor's goal position/speed/acc to the sync write buffer
    for scs_id in SCS_IDS:
        scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, scs_goal_position[index], SCS_MOVING_SPEED, SCS_MOVING_ACC)
        if scs_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

    # Send the single combined packet -- all motors start moving at the same instant
    scs_comm_result = packetHandler.groupSyncWrite.txPacket()
    if scs_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(scs_comm_result))

    # Clear syncwrite parameter storage
    packetHandler.groupSyncWrite.clearParam()

    print("Rotating...")
    # Adjust this delay based on how long a full 360 turn actually takes at SCS_MOVING_SPEED.
    time.sleep(2)
    print("Done -- stopped. Waiting for next keypress.")

    # Flip to the other endpoint for next press (alternates direction each time)
    index = 1 - index

# Close port
portHandler.closePort()