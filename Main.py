from DataProcessing import processing
from Learning import deep_learning
import matplotlib.pyplot as plt

processor = processing()

learning = deep_learning(processor=processor)

history = learning.learn()
mse , mae = learning.get_metrics()

predictions = []
for row in processor.get_original():
    prediction = learning.predict(row)
    predictions.append(prediction)

print(predictions)

# plt.plot(history.history['loss'])
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.title("Training Loss")
# plt.plot((mse))
# plt.scatter(processor.get_original()['Chance of Admit '], range(len(processor.get_original().index)))


# plt.show()


