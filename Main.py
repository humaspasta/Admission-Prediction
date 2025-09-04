from DataProcessing import processing
from Learning import deep_learning

processor = processing()

learning = deep_learning(processor=processor)

learning.learn()



