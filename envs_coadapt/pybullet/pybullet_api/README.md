The files in this directory belong to the pybullet project and were changed to enable the automatic design adaptation of agents during the training process.

You can find the original files here: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym

Sha:
The key changes locate in robot_locomotors.py
Where the class HalfCheetah(WalkerBase) has a method of adapt_xml(self, file, design = None) and allows to take "design" as input for __init__(self, design = None). 

