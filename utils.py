########################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                                     #
########################################################################################################################

import os
#Varibles of the printd Debug function.
DEBUG = True
DEBUG_lvl = 1

def printd(str, lvl=1):
    """
    Function to facilitate the debug.
    It only shows the message if the message lvl is below the DEBUG lvl activated on that moment.
    """
    if DEBUG and DEBUG_lvl >= lvl:
        print("{}".format(str))

def folder_exists(path_save):
    """
    Function that verifies if a folder exist if not it creates the folder.

    :param
    path_save: str (path)
        Path to be verified/created
    :return:
    """
    if not (os.path.exists(path_save)):
        os.mkdir(path_save)