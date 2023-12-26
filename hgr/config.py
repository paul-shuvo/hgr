# from os import path
from os.path import abspath, dirname, join

model_path = join(dirname(abspath(__file__)), 'models', 'hand_landmarker.task') 
print(model_path)