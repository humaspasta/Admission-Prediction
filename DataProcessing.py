import pandas as pd
import os
import kagglehub 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class processing():
    def __init__(self):
        self.path = kagglehub.dataset_download("mohansacharya/graduate-admissions")  
        self.path = os.path.join(self.path , 'Admission_Predict.csv')
        self.data = pd.read_csv(self.path)
        self.features = self.data[: , : self.data.get_loc('Chance of Admit ')]
        self.labels = self.data['Chance of Admit ']
        self.num_features = self.features.shape[1]
        

    


    def preprocess_data(self) -> pd.DataFrame:
        '''
        Drops unecessary columns, applies column transformer for scaling, and encodes catigorical data with one hot encoding
        '''
        self.features = pd.get_dummies(self.features)

        num_features = self.data.select_dtypes(['float64' , 'int64'])
        features_train, features_test, labels_train, labels_test = train_test_split(self.features, self.label, test_size=0.2, random_state=24)
        transformer = ColumnTransformer([
            ('Scale' , StandardScaler(), num_features)
        ], remainder='passthrough')



        features_train_scaled = transformer.fit_transform(features_train)

        features_test_scaled = transformer.fit(features_test)
        return features_train_scaled , features_test_scaled
        

    def get_original(self):
        return self.data
    
    def get_path(self):
        return self.path
    
    




