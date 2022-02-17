from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# print the percentile of feature compared to a specific a dataset
def percentile_score(shap_value,X):
  df = pd.DataFrame(columns=X.columns)
  n_columns = len(X.columns)
  fig, axes = plt.subplots(1,n_columns,figsize=(20,2),constrained_layout=True)
  for name, value, axis in zip(X.columns, shap_value.data, axes):
    score=stats.percentileofscore(X[name],value)
    ax = sns.kdeplot(data=X, x=name, ax=axis)
    x, y = ax.lines[0].get_data()
    width = (np.max(x) - np.min(x))/100
    ax.bar(value,y[x < value], width=width,color='r')
    ax.text(0.8, 0.8, f'PCTL: {score:.2f}%\n{name}: {value:.2f}', color='black',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

# Plot barh
def prob_barh(data):
    with plt.style.context(('ggplot', 'seaborn')):
        plt.figure(figsize=(5, 2)) 
        #print("Probability of no lateral spreading: {}\n Probability of lateral spreading: {}" % format(model.predict_proba(X_train)[idx][0], model.predict_proba(X_train)[idx][1]))
        plt.barh(["No lateral spreading", "Lateral spreading"], data, 0.5,  color=['lightblue', 'orange'])
        plt.text(data[0], "No lateral spreading", "{:.2f}".format(data[0]))
        plt.text(data[1], "Lateral spreading", "{:.2f}".format(data[1]))
        plt.xlim([0,1])
        plt.grid()
        plt.title("Prediction probabilities")
        plt.ylabel("", labelpad=0.1)
        plt.show()
    return None

# plot prediction probability
def pred_prob(data):
    with plt.style.context(('ggplot', 'seaborn')):
        plt.figure(figsize=(8,6))
        sns.displot(data, kde=False, 
                    bins=int(25), color = 'blue',
                    edgecolor='k')
        plt.xlabel('predictive probability')
        plt.ylabel('count')
        plt.show()



# Function to Plot confusion matrix 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Observed',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
