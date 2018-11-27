######################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                   #
######################################################################################################

import os
import re
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

    :param path_save: str (path)
            Path to be verified/created
    :return: nothing
    """
    if not (os.path.exists(path_save)):
        os.mkdir(path_save)

def read_cfg(file):
    appended_text = []
    with open(file) as f:
        for line in f:
            content = line.strip().replace(" ","")
            if len(content) and content[0] != "#" and content[0] != ";":
                if content[0] != "+":
                    key,value = content.split("=",1)
                    appended_text.append("--{}".format(key))
                    if ".." == value[:2]:
                        value=os.path.join(os.path.dirname(__file__),value[3:])
                    appended_text.append(value)
                else:
                    appended_text[-1]="{}{}".format(appended_text[-1],content[1:])
    return appended_text

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')
