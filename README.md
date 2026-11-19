# Ensemble-Methods-Random-Forest-vs-AdaBoost-comparison
## EXPERIMENT - 8

## AIM

Ensemble Methods: Random Forest vs AdaBoost comparison.

---

## ALGORITHM

**Brief Algorithm:**

RandomForest = bagging many trees; AdaBoost = sequential boosting of weak learners 
focusing on hard examples.  

---

## PROGRAM

**Run the following program in a Jupyter Notebook code cell:**

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 

from sklearn.datasets import load_breast_cancer 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score 

X,y = load_breast_cancer(return_X_y=True) 

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3, random_state=42) 

rf = RandomForestClassifier(n_estimators=100, 

random_state=42).fit(Xtr,ytr) 

ab = AdaBoostClassifier(n_estimators=50, random_state=42).fit(Xtr,ytr) 

print('RF accuracy:', round(accuracy_score(yte, rf.predict(Xte)),4)) 

print('AdaBoost accuracy:', round(accuracy_score(yte, 

ab.predict(Xte)),4)) 
```

---

## DETAILED EXPLANATION

1. Load dataset and split. 
2. Train RandomForestClassifier(n_estimators=100) and 
AdaBoostClassifier(n_estimators=50). 
3. Compare test accuracies.
---

---

## OUTPUT

```text
RF accuracy: 0.9591 
AdaBoost accuracy: 0.9406 
```

---

## RESULT

Students compare bagging vs boosting; performance depends on data and hyperparameters.

---

## THEORY  

Bagging reduces variance (averaging), boosting reduces bias (sequential focus on errors). 
