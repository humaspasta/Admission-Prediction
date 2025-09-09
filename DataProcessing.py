import pandas as pd
import os
import kagglehub 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

class processing():
    def __init__(self):
        self.path = kagglehub.dataset_download("mohansacharya/graduate-admissions")  
        self.path = os.path.join(self.path , 'Admission_Predict.csv')
        self.data = pd.read_csv(self.path)
        self.data.drop(columns=['Serial No.'], inplace=True)
        self.copy = copy.deepcopy(self.data)
        last_col = self.data.columns.get_loc('Chance of Admit ')
        self.features = self.data.iloc[: , : last_col]
        self.labels = self.data['Chance of Admit ']
        self.num_features = self.features.shape[1]

        self.features = pd.get_dummies(self.features)

        num_features = self.features.select_dtypes(include=['float64' , 'int64'])

        self.num_features_cols = num_features.columns
        

    
    def preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Drops unecessary columns, applies column transformer for scaling, and encodes catigorical data with one hot encoding
        '''


        features_train, features_test, labels_train, labels_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=24)

        transformer = ColumnTransformer([
            ('Scale' , StandardScaler(), self.num_features_cols)
        ], remainder='passthrough')



        features_train_scaled = transformer.fit_transform(features_train)

        features_test_scaled = transformer.transform(features_test)

        return features_train_scaled , features_test_scaled, labels_train, labels_test, transformer
        

    def get_data(self):
        return self.data
    
    def get_path(self):
        return self.path
    
    def get_num_features_cols(self):
        return self.num_features_cols
    
    




