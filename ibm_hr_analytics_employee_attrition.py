import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
df.drop(columns='EmployeeNumber', inplace=True)
numerical_cols = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_cols = [feature for feature in df.columns if df[feature].dtype == 'O']
categorical_cols.remove('Attrition')

le=LabelEncoder()
df['Attrition']=le.fit_transform(df['Attrition'])

num_pipeline=Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',MinMaxScaler())
    ]
)
cat_pipeline=Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehotencoder', OneHotEncoder())
    ]
)

preprocessor=ColumnTransformer([
    ('num_pipeline', num_pipeline, numerical_cols),
    ('cat_pipeline', cat_pipeline, categorical_cols)
])

x=df.drop(labels=['Attrition'], axis=1)
y=df.Attrition
x_train,x_test, y_train,  y_test=train_test_split(x, y, test_size=0.2, random_state=42)
x_train=preprocessor.fit_transform(x_train)
x_test=preprocessor.transform(x_test)
models={
    'logistic_regression':LogisticRegression(),
    'support_vector':SVC(probability=True),
    'random_forest':RandomForestClassifier(),
    'decision_tree':DecisionTreeClassifier(),
    'KNN':KNeighborsClassifier(),
    'xgboost': xgb.XGBClassifier(n_estimators=1000),
    'naive_bayes':BernoulliNB()
}
for i in range(len(models)):
        model=list(models.values())[i]
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print("Metrics for",list(models.keys())[i])
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))
