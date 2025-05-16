#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# 🔹 Example data
# True labels (1 = fraud, 0 = legit)
y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Model predicted probabilities for positive class (fraud)
y_scores = np.array([0.95, 0.90, 0.80, 0.60, 0.55, 0.45, 0.40, 0.20])

# Apply threshold to get predicted class (here we use 0.5)
y_pred = (y_scores >= 0.5).astype(int)


# In[3]:


# 🔹 Metrics
print("Classification Report:\n")
print(classification_report(y_true, y_pred))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_true, y_scores))


# Macro Precision = (0.67 + 0.60) / 2 = 0.635
# Macro Recall    = (0.50 + 0.75) / 2 = 0.625
# Macro F1        = (0.57 + 0.67) / 2 = 0.62
#  ROC-AUC Score: 0.625
# Measures how well the model separates classes across thresholds.
# 
# 0.5 = random guessing, 1.0 = perfect separation.
# 
# Your model is doing better than random, but there's room to improve.

# In[7]:


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract values
TN, FP, FN, TP = cm.ravel()

# Simple text-based confusion matrix
print("\nConfusion Matrix:")
print("                Predicted")
print("              ┌────────────┬────────────┐")
print("  Actual      │  Legit (0) │ Fraud (1)  │")
print("──────────────┼────────────┼────────────┤")
print(f"Legit (0)     │   TN={TN:<6} │  FP={FP:<6} │")
print(f"Fraud (1)     │   FN={FN:<6} │  TP={TP:<6} │")
print("              └────────────┴────────────┘")


# In[4]:


# 🔹 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[6]:


# 🔹 ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, marker='.', label='ROC curve (AUC = {:.2f})'.format(roc_auc_score(y_true, y_scores)))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()


# In[8]:


# ROC Curve data
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)

# Plot ROC Curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, marker='o', label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
# Add threshold labels
for i, thr in enumerate(thresholds):
    if i % 1 == 0:  # label all points (adjust to %2 or %3 if too crowded)
        plt.annotate(f'{thr:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(5, -10), fontsize=8)

# Formatting
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve with Thresholds')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




