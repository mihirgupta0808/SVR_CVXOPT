#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxopt
import pandas as pd
from numpy import linalg

df = pd.read_csv('./hf.csv')

print(df)



X = df.iloc[:,0:-1].values
Y =  df.iloc[:,-1].values


# In[2]:


from sklearn.utils import shuffle
print(X)
print(type(X))

print(Y)
print(type(Y))

X, Y = shuffle(X, Y)
print(X)
print(Y)


# In[3]:


print(X.shape)
split = int(0.8*X.shape[0])
X_train = X[:split,:]
Y_train = Y[:split]
X_test = X[split:,:]
Y_test = Y[split:]
print(X_train)


# In[4]:


import numpy
from cvxopt import matrix
from cvxopt import solvers


# In[5]:


def linear(x1, x2):
    return np.dot(x1, x2)

def polynomial(x1, x2, p=2):
    return (1 + np.dot(x1, x2)) ** p

def gaussian(x1,x2,sigma = 5):
    return np.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

def kernel(x1,x2,ktype = "linear",p = 2,sigma = 5):
    if ktype == "gaussian":
        return gaussian(x1,x2,sigma)
    if ktype == "polynomial" :
        #print("poly")
        return polynomial(x1,x2,p)
    #print("lin")
    return linear(x1,x2)


# # predict Function gives the predictions on data for MEDV

# In[6]:


def predict(X_train ,Y_train ,X_test,Y_test,C = 6 ,epsilon = 5 , ktype = "polynomial" , p = 2 , sigma = 5 ):
    n_samples = X_train.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] =kernel(X_train[i],X_train[j],ktype,p,sigma) 
            #np.dot(X_train[i],X_train[j])
    #print(K)
    K1 = np.vstack((K,-K))
    K2 = np.vstack((-K,K))

    P = np.hstack((K1, K2))
    #print(2*n_samples)

    #print(P.shape)
    P = matrix(P)
    #print(P)
    #epsilon = 5 
    q1 = -epsilon + Y_train
    #print(type(q1))
    #print(q1.shape)
    q2 = -epsilon - Y_train
    q = np.hstack((q1,q2))
    #print(q.shape)
    #print(type(q))
    ## 2mX1
    q = matrix(q)
    #print(q)
    #  1 X 2m A 
    A1  = np.ones(n_samples)
    A = np.hstack((A1,-A1))
    #print(A,type(A),A.shape)
    A = matrix(A)
    A = A.trans()
    #print(A)
    #print(A.size)
    b = 0 
    G1 = np.identity(2*n_samples)
    G2 = -np.identity(2*n_samples)
    G = np.vstack((G1,G2))
    #print(G.shape,G)
    G = matrix(G)
    #C = 6
    Cmat = C*np.ones(2*n_samples)
    Omat = np.zeros(2*n_samples)
    h = np.hstack((Cmat,Omat))
    #print(h.shape,h)
    # h is 4mx1
    h = matrix(h)

    b = matrix([0.0])
    #print(b.size)
    b = b.trans()
    #print(b.size)
    from cvxopt import solvers
    sol = solvers.qp(P,q,G,h,A,b)
    #print(X_train[0])
    #print(X_train)
    ans = sol['x']
    #print(ans.size)
    np_ans =  np.ravel(ans)
    #print(np_ans.shape)
    half = np_ans.shape[0]//2
    #print(half)
    a_s = np_ans[0:half]
    a_primes = np_ans[half:]
    #print(a_s.shape)
    #print(a_primes.shape)
    sum = 0 
    for i in range(n_samples):
        sum += Y_train[i] - epsilon
        for j in range(n_samples):
            sum-= (a_s[j] - a_primes[j])*kernel(X_train[i],X_train[j])
    b = sum / n_samples
    #print(b)
    pred_test = np.zeros(Y_test.shape)
    #print(Y_test.shape)
    #print(pred_test.shape)
    test_samples = Y_test.shape[0]
    #print(test_samples)
    for i in range(test_samples):
        val = b
        for j in range(n_samples):
            val += (a_s[j] - a_primes[j])*kernel(X_test[i],X_train[j])
        pred_test[i] = val 


    #print(pred_test , pred_test.shape)
    #print(Y_test ,Y_test.shape)
    
    return pred_test


# In[7]:


def MSE(predictions,Y):
    return np.square(predictions - Y).mean()
    


# # Varying C for Linear Kernel

# In[8]:


#lintest = predict(X_train ,Y_train ,X_test,Y_test,ktype = "linear",epsilon = 1,C = 0.00000000000000000001,p=2)
#lintrain = predict(X_train ,Y_train ,X_train,Y_train,ktype = "linear",epsilon = 1,C = 0.00000000000000000001,p=2)
#print(MSE(lintest,Y_test))
#print(MSE(lintrain,Y_train))
#X_train ,Y_train ,X_test,Y_test,C = 6 ,epsilon = 5 , ktype = "polynomial" , p = 2
def MSEttest( C,ktype = "linear",epsilon = 1,p = 2 ,sigma = 5):
    lintrain = predict(X_train ,Y_train ,X_train,Y_train,C,epsilon,ktype,p,sigma)
    lintest = predict(X_train ,Y_train ,X_test,Y_test,C,epsilon,ktype,p,sigma)
    
    return MSE(lintrain,Y_train),MSE(lintest,Y_test)

from matplotlib import pyplot as plt
#Cs = [ 0.00000000000000000001 ,  0.0000000000000001 ,0.000000000001,0.0000000001,0.00000001,0.000001,0.0001,0.001,0.01,0.1,1,5,10]
#Cs = [1,5]
#Cs = [1,10,40,120,200,300,1000,3000]
Cs = [1000,2000,5000,10000,15000,25000]
trains=[]
tests=[]
for C in Cs:
    train,test = MSEttest(C)
    trains.append(train)
    tests.append(test)

print(train.shape)
print(test.shape)
#with plt.style.context('seaborn-whitegrid'):
    
    
plt.scatter(Cs,trains,label='train',c='blue')
plt.scatter(Cs,tests,label='test',c='red')
    
    
    
plt.xlabel('C')
plt.ylabel('MSE')
plt.legend()
plt.show()







# # Varying Kernel Type

# In[9]:


trains=[]
tests=[]
#C = 0.000001
C = 0.1
variants = ["linear","polynomial","sklearn's SVR"]
vals = [1,2,3]
for variant in variants:
    if variant == "sklearn's SVR":
        break
    train,test = MSEttest(C,ktype = variant)
    trains.append(train)
    tests.append(test)




from sklearn.svm import SVR

clf = SVR(C=C, epsilon=1)
clf.fit(X_train, Y_train)
trains.append(MSE(clf.predict(X_train),Y_train))
tests.append(MSE(clf.predict(X_test),Y_test))


print("trains," , len(trains))

#MSE(lintrain,Y_train),MSE(lintest,Y_test)

'''
from sklearn.svm import SVR
import numpy as np
clf = SVR(C=1.0, epsilon=5)
clf.fit(X_train, Y_train)
predsk = clf.predict(X_test)
print(predsk)


'''
    
plt.xticks(vals, variants)
plt.scatter(vals,trains,label='train',c='blue')
plt.scatter(vals,tests,label='test',c='red')
plt.xlabel('kernel/method')
plt.ylabel('MSE')
plt.show()


# # Varying C for Sklearn's SVR

# In[10]:


#Cs = [ 0.00000000000000000001 ,  0.0000000000000001 ,0.000000000001,0.0000000001,0.00000001,0.000001,0.0001,0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,50,60,70,80,90,100,120,200]
Cs = [200,300,400,500,600,700,800,1000,2000,3000,4000,5000,20000,100000,400000,700000,1000000]
trains = []
tests  = []
from sklearn.svm import SVR

for C in Cs:
    clf = SVR(C=C, epsilon=1)
    clf.fit(X_train, Y_train)
    trains.append(MSE(clf.predict(X_train),Y_train))
    tests.append(MSE(clf.predict(X_test),Y_test))
    
    
    
plt.scatter(Cs,trains,label='train',c='blue')
plt.scatter(Cs,tests,label='test',c='red')
    
    
    
plt.xlabel('C')
plt.ylabel('MSE')
plt.legend()
plt.show()














# # Varying Sigma For Gaussian Kernel

# In[11]:


#C = 0.000001
C = 0.1
#sigs = [0.01,0.03,0.09,0.27,0.81,0.1,0.3,0.9,2.7,6,9]
sigs = [1,10,20,30,100,1000,2000,3000,4000,5000]
trains=[]
tests=[]
for sig in sigs:
    train,test = MSEttest(C,ktype = "gaussian",sigma = sig)
    trains.append(train)
    tests.append(test)

print(train.shape)
print(test.shape)
#with plt.style.context('seaborn-whitegrid'):
    
    
plt.scatter(sigs,trains,label='train',c='blue')
plt.scatter(sigs,tests,label='test',c='red')
    
    
    
plt.xlabel('sigma')
plt.ylabel('MSE')
plt.legend()
plt.show()





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




