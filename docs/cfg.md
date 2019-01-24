# Configuration files .CFG
The files with extension .cfg are files that have configurations to the execution of the reinforcement learning algorithm. These files are an alternative that facilitates the execution and debug of the training/testing of the agents in the chosen environments. Instead of type long codes in the terminal in combination with execution of the script, we write these configurations in the file .cfg that have the same name as the script to be executed, for example, Base_agent.cfg for the script Base_agent.py. Under the hood, what the script does is to read the cfg file and transform each valid line of code in terminal commands that, once converted, are sent to Argparser before initialization. Hence, one line of code env = Doom is converted in --env Doom and it is sent to the main method of the executed script. An important fact to watch in the cfg files is that: **Case some terminal command is sent jointly with the execution of the script, the entire configuration of the simulation will be done by the terminal commands, thus, the .cfg file will be ignored and any parameter that was not sent via terminal will have its default value assigned.** 

## Writing the .cfg files
- Each line starting with the characters **#** or **;** are treated as comments.
- Each line starting with the character **+** are a continuation from the previous line. 
- The lines are case insensitive
- Each line should have only one pair of key = value.
- If an argument is not specified in the file .cfg and is necessary to the execution of the desired task, its default value will be assigned. This is done to avoid the necessity of constantly have to write the parameters that have their values used often in the simulation. For default values of each variable
see the [Documentation](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/doc.md) and for examples see the section [Examples](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/examples.md).
- Relative paths in relation the main directory can be inserted using **..** before the path. For example, to access the DoomScenarios directory we can use "../DoomScenarios".
- The commands are read in sequence, thus, in case of repeated commands, only the last will be valid. 
