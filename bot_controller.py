import sys
import os
import math
sys.path.insert(0, os.path.abspath('.'))

from lib.interface import Interface
from lib.dobot import Dobot

class BotController:
    #Hier '<xxx>' ersetzen mit spezifischem Device-String: 'COM3' für Windows-Laborrechner
    #Bei Linux herausfinden mit 'ls /dev/tty.*' )
    def __init__(self):
        self.bot = Dobot('COM3')

    def __move_bot_to_paper(self):
        bot = self.bot

        print('Unlock the arm and place it on the middle of the paper')
        input("Press enter to continue...")

        center = bot.get_pose()
        print('Center:', center)

        bot.move_to_relative(0, 0, 10, 0)
        print('Ready to draw')

        bot.move_to_relative(0, 0, -10, 0)

        bot.interface.set_continous_trajectory_params(200, 200, 200)

        return center

    def homing(self):
        bot = self.bot
        print('Bot status:', 'connected' if bot.connected() else 'not connected')

        print('Homing')
        bot.home()

    def draw_circle (self):
        bot = self.bot

        center = self.__move_bot_to_paper()
        path = []
        steps = 24
        scale = 50
        for i in range(steps + 2):
            x = math.cos(((math.pi * 2) / steps) * i)
            y = math.sin(((math.pi * 2) / steps) * i)

            path.append([center[0] + x * scale, center[1] + y * scale, center[2]])
        bot.follow_path(path)

        # Move up and then back to the start
        bot.move_to_relative(0, 0, 10, 0)
        bot.slide_to(center[4], center[5], center[6], center[7])


    def draw_triangle (self):
        bot = self.bot
        
        self.__move_bot_to_paper()
        
        bot.move_to_relative(40, 80 , 0, 0)
        bot.move_to_relative(40, -80 , 0, 0)
        bot.move_to_relative(-80, 0, 0, 0)

    def draw_square (self):
        bot = self.bot
        
        self.__move_bot_to_paper()

        bot.move_to_relative(0, 80 , 0, 0)
        bot.move_to_relative(80, 0 , 0, 0)
        bot.move_to_relative(0, -80 , 0, 0)
        bot.move_to_relative(-80, 0, 0, 0)