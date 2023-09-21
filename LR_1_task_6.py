import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

classifier_svm = SVC()
classifier_svm.fit(X, y)

classifier_g = GaussianNB()
classifier_g.fit(X, y)


# виведення показників
def print_stats(classifier, X, y):
    num_folds = 3
    stats = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    print(classifier.__class__.__name__)

    for item in stats:
        item_val = cross_val_score(classifier, X, y, scoring=item, cv=num_folds)
        print(item + ": " + str(round(100 * item_val.mean(), 2)) + "%")

    print()


print_stats(classifier_svm, X, y)
print_stats(classifier_g, X, y)

visualize_classifier(classifier_svm, X_test, y_test)
visualize_classifier(classifier_g, X_test, y_test)
