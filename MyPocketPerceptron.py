import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
class MyPocketPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.w_ = []
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        self.w_ = self._perceptron_learning(X, y)
        return self
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        y = []
        for x in X :
          result = MyPocketPerceptron._classify(x, self.w_)
          y.append(result)
        return y
    @staticmethod
    def _classify(x, w):
        y = np.dot(w, np.insert(x,0,1).transpose())
        target = 1.0 if (y > 0) else 0.0
        return target

    def _perceptron_learning(self, X, y):   
        #initializing weight vector with additional column for bias
         w = np.zeros(shape=(1, X.shape[1]+1))
         #initalizing learning rate, 0.1 is common
         learning_rate=0.1
         x_nums = len(X)
         #keeping track of number correct weight in every iteration
         best_accurate=0
         #nitializing the pocket = best performing weights
         best_pocket_weight=w.copy()
         for attempt in range(self.max_iter):
          #number of errors
           missclassification_count=0
            #going through each feature, and combining both input and label into one
           for x_i, target in zip(X, y):
             # Add bias term
             x_i_init = np.insert(x_i, 0, 1) 
             #calculated weighted sum with initialized input with bias
             weighted_sum = np.dot(w, x_i_init)  
             #applying logical sign function 
             if weighted_sum >=0: 
               output=1
             else:
               output=-1
              #condition where the y and ^y are not =
             if output!= target: 
            #update weight, when both y and y^ are same, difference is zero, and w stays the same
              w= w + learning_rate*(target-output)* x_i_init.reshape(1, -1) 
              #update error count as weight is incorrect
              missclassification_count+=1

              #calculate sum of all predictions, which is an array of weighted_sum for all features, but only those that equal the correct label preds==y)
              predictions = [self._classify(x, w) for x in X]
              accurate = np.sum(predictions == y)
              correct = (x_nums-missclassification_count)/x_nums
              #update pocket with number of total correct, and most correct
              if accurate>best_accurate:
                best_accurate=accurate
                best_pocket_weight=w.copy()
                #return if equals exactly number of features, which means all are accurate
                if best_accurate==len(X):
                    return best_pocket_weight
            #otherwise break when error count is lowest
           if missclassification_count==0:
              break
          
         return best_pocket_weight