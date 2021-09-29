import os

from utilities import generateExternalFunction


pathMain = os.getcwd()

pathExample = os.path.join(pathMain, 'example')
pathOpenSimModel = os.path.join(pathExample, 'generic_model.osim')
pathID =  os.path.join(pathMain, 'InverseDynamics')


generateExternalFunction(pathOpenSimModel, pathExample, pathID)
