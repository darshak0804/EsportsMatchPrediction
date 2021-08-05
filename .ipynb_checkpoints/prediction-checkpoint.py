import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv('./match_dataset.csv')

trimmed_df = df[['my_trophies', 'opponent_trophies', 'my_deck_elixir', 'op_deck_elixir',
         'my_troops', 'my_buildings', 'my_spells', 'op_troops', 'op_buildings', 'op_spells',
         'my_commons', 'my_rares', 'my_epics', 'my_legendaries',
         'op_commons', 'op_rares', 'op_epics', 'op_legendaries']]

class_col = df[['my_result']]



from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

labelencoder_y = preprocessing.LabelEncoder()
encoded_class_col = labelencoder_y.fit_transform(class_col)

features = trimmed_df.to_numpy()
TrainFeatures, TestFeatures, TrainClass, TestClass = train_test_split(features,
                                                                      encoded_class_col,
                                                                      test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier

print(len(features))

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(TrainFeatures, TrainClass)

prediction = classifier.predict(TestFeatures)

print("Accuracy: ", accuracy_score(TestClass, prediction)) 

mytest = classifier.predict(np.array([[2513,2477,3.625,3.875,5,1,2,7,0,1,4,1,2,1,1,2,4,1]]))
print(mytest)

# sns.scatterplot(, TrainClass)
# plt.show()