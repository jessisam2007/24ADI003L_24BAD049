print("JESSICA SAM 24BAD049")
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
patient_df = pd.read_csv(
    r"/Users/jessicasam/Downloads/diabetes.csv"
)
print(patient_df.head())
print(patient_df.info())
print(patient_df.isnull().sum())
 
plt.hist(patient_df["Glucose"], bins=20)
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.show()
 
sns.boxplot(x=patient_df["Age"])
plt.title("Age Distribution of Patients")
plt.show()
