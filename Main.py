from DataProcessing import processing
from Learning import deep_learning
import matplotlib.pyplot as plt
import numpy as np

processor = processing()

learning = deep_learning(processor=processor)

history = learning.learn()
mse , mae = learning.get_metrics()

predictions = []
data = processor.get_data()
rows = data.iloc[1: , :]
rows = rows.drop(columns=['Chance of Admit '])
# np_arr = np.array(rows.iloc[1])
# reshaped = np.expand_dims(np_arr, axis=0)
# print(np_arr)
# print(learning.predict(reshaped))
for row in range(len(rows)):
    np_arr = np.array(rows.iloc[row])
    reshaped = np.expand_dims(np_arr, axis=0)
    prediction = learning.predict(reshaped)
    predictions.append(prediction)

print(predictions)
plt.scatter(predictions, range(len(predictions)))
plt.show()
plt.scatter(processor.labels, range(len(processor.labels)))
plt.show()

