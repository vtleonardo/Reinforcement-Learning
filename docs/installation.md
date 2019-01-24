# Instalation
The entire code was written and tested in python 3.6 on windows 10. To execute the code, the following packages are needed:
````
Tensorflow (cpu ou gpu)
Keras
Pandas
Imageio
OpenCV
Matplotlib
OpenAI Gym
ViZDoom
````

For the installation of the packages listed above, it is recommended to create a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) with [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). With the virtual environment activated, the installation of the packages can be done with the following commands in the terminal:

To install the CPU vertion of the Tensorflow
````
conda install tensorflow
````
To install the GPU vertion of the Tensorflow
````
conda install tensorflow-gpu
````
For the rest of the packages:
````
conda install keras
conda install pandas
conda install imageio
conda install opencv
conda install matplotlib
````
For the installation of the open ai gym together with the games of Atari 2600 on windows, use the following commands:
````
pip install gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
````
Para mais detalhes sobre a execução dos jogos de atari no windows, consultar esse [link](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows).

For more details about the execution of the atari games on windows, see this [link](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#windows_bin).

Once all packages are installed and the virtual environment is configured, just clone or download this repository and execute the Base_agent.py for the training of an agent with the reinforcement learning algorithms DQN or DRQN.
