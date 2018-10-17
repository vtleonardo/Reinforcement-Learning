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
    if not (os.path.exists(path_save)):
        os.mkdir(path_save)