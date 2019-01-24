# Maps
Together with the scripts that compose this repository, it was created two maps for the ViZDoom environment. For the development of these maps, it was thought of an environment that could imitate some problem of mobile robotics related to its navigation part. The problem chosen was a mobile robot that needs to navigate back to its battery recharging platform. Thus, were created maps in which the agent aims to find his battery recharging platform as fast as possible. Therefore, the agent is encouraged to not waste time performing actions in one place and also to avoid collisions of any kind with the environment.

<p align="center">
 <img src="https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/images/mapa-2D.png" height="100%" width="100%">
</p>

The figure above shows the top view of both maps: labyrinth and labyrinth test. The red circles indicate the spots where the agent can start an episode. Each map consists of two rooms separated by a 'curved' corridor and, in each episode, the agent is placed randomly in some of the circles looking in some direction that is also random. The green rectangle (in the upper right corner of the first map and the lower left corner of the second map) shows the position of the battery recharging platform the agent has to reach. Although both maps look alike, having just a difference in the rotation, in the second map, the agent can not start in a position where it can immediately see the platform due to a wall. This map, called labyrinth_test, was created to test the transfer learning capabilities of a trained agent in the first map called labyrinth.

<p align="center">
 <img src="https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/images/mapa-3D.png" height="100%" width="100%">
</p>

The figure above demonstrates the maps by the agent's view. During the development of the maps, the roof was placed high above so that it did not appear in the agent's view. This was done to try to approach the vision of a small mobile robot navigating in a room. In addition, each room has different walls textures so the agent can identify where it is located.

# Modeling as an MDP (Markov Decision Process) 
The modeling for both maps as an MDP (Markov Decision Process) can be seen below:
- **States/Observations:** The sequence of images obtained from the ViZDoom platform concatenated to create a volume.
- **Actions**: 
  - Move Forward
  - Turn the camera toright and left
- **Rewards:**
  - \-0.001 for each lived game frame, encouraging the agent to locate the battery recharging platform as fast as possible.
  - \-0.01 for every 5 game frames that the agent stays in the same position, encouraging the agent not to stand still performing actions and consequently spending battery in vain, in the same location.
  - \-0.1 by collision on any wall or object.
  - \+10 when reaching the main goal (battery recharging platform).
 - **End of episode:** When the agent reaches the battery recharging platform or time runs out (3000 game frames).
 
   **Each [frame_skip] number of game frame is equal to one simulation frame. For example, 3000 game frames are equal to 750 simulation frames using the frame skip equals 4, this is because the same action is executed 4 times (4 game frames) inside the algorithm. **
