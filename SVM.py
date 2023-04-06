import numpy as np
from kernels import linear_kernel
from kernels import poly_kernel
from kernels import rbf_kernel
from kernels import sigmoid_kernel
from model_functions import _select
from model_functions import _bounds



class SVM:
    def __init__(self, kernel='linear', C=0.1, degree=3, gamma=0.01, \
                 coef0=1.0, theta = 1.0, coef1 = 0.0 ,tol=1e-3, n_epochs=100, verbose = True):

        #PARAMETERS INITIALIZATION
        self.kernel = kernel            #kernel
        self.C = C                      #C parameters
        self.degree = degree            #degree of polynomial kernel
        self.coef0 = coef0              #coef of poly kernel
        self.theta = theta              #parameter of sigmoid kernel
        self.coef1 = coef1              #coef of sigmoid kernel
        self.tol = tol                  #tolerance for early stopping
        self.n_epochs = n_epochs        #number of epochs
        self.gamma = gamma              #parameter of rbf kernel
        self.verbose = verbose          #check to have output while training

    def fit(self, X, y):

            self.X = X      # data
            self.y = y      # labels
            n_samples, n_features = X.shape     
            self.alpha = np.random.uniform(low=0, high=self.C, size=(n_samples,))  #initialization of lagrange multipliers alphas
            self.b = 0      # b initialization
            if self.gamma == 'scale': self.gamma = 1.0 / (n_samples)

            self.K = self._kernel(self.X,self.X)    #kernel generation from data
              

            #SOLVER (SMO)
            epoch = 0
            while epoch < self.n_epochs:
                if self.verbose == True and epoch % 10 == 0: print("Number of epochs:", epoch)
                alpha_changed = 0

                if self.solver == "SMO": 
                  for i in range(n_samples):

                      Ei = self._margin(i) - y[i]           #select first error
                      if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or (y[i] * Ei > self.tol and self.alpha[i] > 0): #tolerance check
                          j = _select(i)                    #second is picked at random
                          Ej = self._margin(j) - y[j]       #select second error

                          ai_old, aj_old = self.alpha[i], self.alpha[j]
                          L, H = _bounds(self.C, ai_old, aj_old, y[i], y[j])  #compute L and H bounds
                          if L == H: continue               #skip loop if condition
                          
                          eta = 2 * self.K[i,j] - self.K[i,i] - self.K[j,j]
                          if eta == 0: continue             #skip loop if condition

                          self.alpha[j] = aj_old - y[j] * (Ei - Ej) / eta       #compute first alpha
                          self.alpha[j] = np.clip(self.alpha[j], L, H)          #clip alpha between bounds

                          if abs(self.alpha[j] - aj_old) < 1e-5: continue       #skip loop if condition

                          self.alpha[i] = ai_old + y[i] * y[j] * (aj_old - self.alpha[j])   #compute second alpha

                          #b computing and upgrading
                          b1 = self.b - Ei - y[i] * (self.alpha[i] - ai_old) * self.K[i,i] - y[j] * (self.alpha[j] - aj_old) * self.K[i,j]
                          b2 = self.b - Ej - y[i] * (self.alpha[i] - ai_old) * self.K[i,j] - y[j] * (self.alpha[j] - aj_old) * self.K[j,j]
                          if 0 < self.alpha[i] < self.C:      self.b = b1
                          elif 0 < self.alpha[j] < self.C:    self.b = b2
                          else:                               self.b = (b1 + b2) / 2
                          alpha_changed += 1    #check if alpha has been upgraded in this epoch

                if alpha_changed == 0:          #early callback if no upgrade of parameters
                    break
            
            #save support vectors info
            self.support_vectors = np.where(self.alpha > 0)[0]
            self.support_vector_labels = y[self.support_vectors]
            self.support_vector_alphas = self.alpha[self.support_vectors]
            self.x_support_vectors = self.X[self.support_vectors]

    #call appropriate kernel function
    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            K = linear_kernel(x1,x2)
        elif self.kernel == 'poly':
            K = poly_kernel(x1,x2, self.degree, self.coef0)    
        elif self.kernel == 'rbf':
            K = rbf_kernel(x1, x2,  self.gamma)
        elif self.kernel == 'sigmoid':
            K = sigmoid_kernel(x1, x2, self.theta, self.coef1)
        else:
            raise ValueError(f'Unsupported kernel: {self.kernel}')
        return K

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        K = self._kernel(X,self.x_support_vectors)
        for i in range(X.shape[0]):
            s = self.b +np.sum(self.support_vector_alphas * self.support_vector_labels * K[i,:])
            y_pred[i] = np.sign(s)
        return y_pred

    def _margin(self, i):
        func_i = self.b + np.sum(self.alpha * self.y * self.K[i,:])     #compute margin function for each sample
        return func_i

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)