import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Original data/Technographic Data .csv')
df.drop(columns=['Ticker'], inplace=True)
df["First Seen At"] = pd.to_datetime(df["First Seen At"])
df["Last Seen At"] = pd.to_datetime(df["Last Seen At"])
df["Usage_Duration_Days"] = (df["Last Seen At"] - df["First Seen At"]).dt.days
df.drop(columns=["First Seen At", "Last Seen At"], inplace=True)

label = LabelEncoder()
df["Website Domain"] = label.fit_transform(df["Website Domain"])
df["Technology Name"] = label.fit_transform(df["Technology Name"])
df["Technology ID"] = label.fit_transform(df["Technology ID"])
df["Behind Firewall"] = df["Behind Firewall"].astype(int)

X = df.drop(columns=["Behind Firewall"])
y = df["Behind Firewall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("Y_train.csv", index=False)
y_test.to_csv("Y_test.csv", index=False)

features = ['Website Domain', 'Technology Name', 'Technology ID', 'Usage_Duration_Days']
target = 'Behind Firewall'

scaler = StandardScaler()
scaler.fit(X_train[features])

X_train_scaled = scaler.transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

sns.countplot(x="Behind Firewall", data=df)
plt.title("Distribution of Behind Firewall Classes")
plt.xlabel("Behind Firewall")
plt.ylabel("Count")
plt.show()

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in models.items():
    print(f"\n** Training {name} **")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred))
