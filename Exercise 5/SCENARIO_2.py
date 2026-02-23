print("24BAD049 - JESSICA SAM B")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/Users/jessicasam/Downloads/train_u6lujuX_CVtuZ9i (1).csv")

features = ['ApplicantIncome', 'LoanAmount', 'Credit_History',
            'Education', 'Property_Area']

X = df[features].copy()
y = df['Loan_Status']

X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].median())
X['ApplicantIncome'] = X['ApplicantIncome'].fillna(X['ApplicantIncome'].median())
X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])
X['Education'] = X['Education'].fillna(X['Education'].mode()[0])
X['Property_Area'] = X['Property_Area'].fillna(X['Property_Area'].mode()[0])

le_edu = LabelEncoder()
le_prop = LabelEncoder()
le_target = LabelEncoder()

X['Education'] = le_edu.fit_transform(X['Education'])
X['Property_Area'] = le_prop.fit_transform(X['Property_Area'])
y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

depths = range(1, 11)
train_acc = []
test_acc = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

best_depth = depths[np.argmax(test_acc)]

dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Best Depth:", best_depth)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=X.columns, class_names=le_target.classes_, filled=True)
plt.show()

importances = dt.feature_importances_

plt.figure()
plt.bar(X.columns, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()

plt.figure()
plt.plot(depths, train_acc)
plt.plot(depths, test_acc)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy (Overfitting Analysis)")
plt.show()

shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)

shallow_tree.fit(X_train, y_train)
deep_tree.fit(X_train, y_train)

print("Shallow Tree Test Accuracy:",
      accuracy_score(y_test, shallow_tree.predict(X_test)))
print("Deep Tree Test Accuracy:",
      accuracy_score(y_test, deep_tree.predict(X_test)))

print("Feature Importances:")
for feature, importance in zip(X.columns, importances):
    print(feature, ":", importance)
