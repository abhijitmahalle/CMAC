# # This code is a program for both Discrete and Continuous CMAC. 
# # Mode argument can be used to use either Discrete or Continuous CMAC.
from time import time
import numpy as np
import math
import matplotlib.pyplot as plt

#Defining function that creates a map which defines the architecture of CMAC
def initialize_model(X,num_assoc,num_weights):
    overlap = num_assoc - 1
    x = np.linspace(np.min(X),np.max(X),num_weights-overlap)
    table = np.zeros((len(x),num_weights))
    
    for i in range(len(x)):
        table[i][i:i+overlap+1] = 1
    
    Weights = np.ones(num_weights)
    model = [x,table,Weights,num_assoc]
    return model

#Function that trains the model based on the training data
def train_model(model,x_train,y_train,max_err,mode):
    start_time = time()
    input_idx = np.zeros((len(x_train),2))
    
    for i in range(len(x_train)):
        if x_train[i] > model[0][-1]:
            input_idx[i][0] = len(model[0]) - 1
        else:
            if x_train[i] < model[0][0]:
                input_idx[i][0] = 0
            else:
                idx = np.where(model[0] <= x_train[i])
                input_idx[i][0], = np.where(model[0] == np.max(model[0][idx]))
                if (model[0][int(input_idx[i][0])] != x_train[i]) & mode:
                    input_idx[i][1] = input_idx[i][0] + 1
    
    lr = 0.05
    err = np.inf
    itr = 0
    count = 0
    
    while (err>max_err) & (2*count<=itr):
        old_err = err
        itr = itr + 1
        
        for i in range(len(input_idx)):
            if input_idx[i][1] == 0:
                weights_idx = np.nonzero(model[1][int(input_idx[i][0])])
                output = np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2]))
                err = lr*(y_train[i]-output)/model[3]
                model[2][weights_idx] = model[2][weights_idx] + err
            else:
                lower_dist = x_train[i] - model[0][int(input_idx[i][0])]
                upper_dist = model[0][int(input_idx[i][1])] - x_train[i]
                weights_idx = np.nonzero(model[1][int(input_idx[i][0])])
                weights_idx2 = np.nonzero(model[1][int(input_idx[i][1])])
                output = (upper_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2])) + (lower_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][1])],model[2]))
                err = lr*(y_train[i]-output)/model[3]
                model[2][weights_idx] = model[2][weights_idx] + (upper_dist/(lower_dist+upper_dist))*err
                model[2][weights_idx2] = model[2][weights_idx2] +  (lower_dist/(lower_dist+upper_dist))*err
        
        num = 0
        denom = 0
        
        for i in range(len(input_idx)):
            if input_idx[i][1] == 0:
                output = np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2]))
                num = num + abs(y_train[i] - output)
                denom = denom + y_train[i] + output
            else:
                lower_dist = x_train[i] - model[0][int(input_idx[i][0])]
                upper_dist = model[0][int(input_idx[i][1])] - x_train[i]
                output = (upper_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2])) + (lower_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][1])],model[2]))
                num = num + abs(y_train[i] - output)
                denom = denom + y_train[i] + output
        
        err = abs(num/denom)
        
        if abs(old_err - err) < 0.00001:
            count = count + 1
        else:
            count = 0
    
    itr = itr - count
    num = 0
    denom = 0
    outputs = []
    
    for i in range(len(input_idx)):
        if input_idx[i][1] == 0:
            output = np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2]))
            num = num + abs(y_train[i] - output)
            denom = denom + y_train[i] + output
            outputs.append(output)
        else:
            lower_dist = x_train[i] - model[0][int(input_idx[i][0])]
            upper_dist = model[0][int(input_idx[i][1])] - x_train[i]
            output = (upper_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2])) + (lower_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][1])],model[2]))
            num = num + abs(y_train[i] - output)
            denom = denom + y_train[i] + output
            outputs.append(output)
    
    Final_err = abs(num/denom)
    idx = np.argsort(x_train)
    x_train = x_train[idx]
    outputs = np.array(outputs)
    outputs = outputs[idx]
    y_train = y_train[idx]
    end_time = time()
    time_diff = end_time - start_time
    return model,itr,Final_err,time_diff

#Function that tests the model and returns the accuracy.
def test_model(model,x_test,y_test,mode):
    input_idx = np.zeros((len(x_test),2))
    
    for i in range(len(x_test)):
        if x_test[i] > model[0][-1]:
            input_idx[i][0] = len(model[0]) - 1
        else:
            if x_test[i] < model[0][0]:
                input_idx[i][0] = 0
            else:
                idx = np.where(model[0] <= x_test[i])
                input_idx[i][0], = np.where(model[0] == np.max(model[0][idx]))
                if (model[0][int(input_idx[i][0])] != x_test[i]) & mode:
                    input_idx[i][1] = input_idx[i][0] + 1
    
    num = 0
    denom = 0
    outputs = []
    
    for i in range(len(input_idx)):
        if input_idx[i][1] == 0:
            output = np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2]))
            num = num + abs(y_test[i] - output)
            denom = denom + y_test[i] + output
            outputs.append(output)
        else:
            lower_dist = x_test[i] - model[0][int(input_idx[i][0])]
            upper_dist = model[0][int(input_idx[i][1])] - x_test[i]
            output = (upper_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][0])],model[2])) + (lower_dist/(lower_dist+upper_dist))*np.sum(np.multiply(model[1][int(input_idx[i][1])],model[2]))
            num = num + abs(y_test[i] - output)
            denom = denom + y_test[i] + output
            outputs.append(output)
    
    err = abs(num/denom)
    accuracy = 100 - err
    idx = np.argsort(x_test)
    x_test = x_test[idx]
    outputs = np.array(outputs)
    outputs = outputs[idx]
    y_test = y_test[idx]
    return accuracy


X = np.linspace(-50,50,100)
np.random.shuffle(X)
Y = X**2
x_train = X[:70]
y_train = Y[:70]
x_test = X[70:]
y_test = Y[70:]
acc_disc = []
acc_cont = []
t_disc = []
t_cont = []

for i in np.linspace(1,34,34):
    model = initialize_model(X,int(i),35)
    model,itr,Final_err,time_diff = train_model(model,x_train,y_train,0.2,0)
    acc_disc.append(test_model(model,x_test,y_test,0))
    t_disc.append(time_diff)

    model = initialize_model(X,int(i),35)
    model,itr,Final_err,time_diff = train_model(model,x_train,y_train,0.2,1)
    acc_cont.append(test_model(model,x_test,y_test,1))
    t_cont.append(time_diff)

plot1 = plt.figure(1)
plt.plot(np.linspace(1,34,34),acc_disc)
plt.title('Accuracy vs Overlap (Discrete CMAC)')
plt.xlabel('Overlap')
plt.ylabel('Accuracy (%)')

plot2 = plt.figure(2)
plt.plot(np.linspace(1,34,34),acc_cont)
plt.title('Accuracy vs Overlap (Continuous CMAC)')
plt.xlabel('Overlap')
plt.ylabel('Accuracy (%)')

plot3 = plt.figure(3)
plt.plot(np.linspace(1,34,34),t_disc)
plt.title('Time to Convergence vs Overlap (Discrete CMAC)')
plt.xlabel('Overlap')
plt.ylabel('Time (s)')

plot4 = plt.figure(4)
plt.plot(np.linspace(1,34,34),t_cont)
plt.title('Time to Convergence vs Overlap (Continuous CMAC)')
plt.xlabel('Overlap')
plt.ylabel('Time (s)')

plot5 = plt.figure(5)
plt.plot(np.linspace(1,34,34),acc_disc,'b', label = 'Discrete CMAC')
plt.plot(np.linspace(1,34,34),acc_cont,'r', label = 'Continuous CMAC')
plt.legend(loc="upper right")
plt.title('Accuracy vs Overlap')
plt.xlabel('Overlap')
plt.ylabel('Accuracy (%)')
plt.savefig('Accuracy vs Overlap.png')

plot6 = plt.figure(6)
plt.plot(np.linspace(1,34,34),t_disc,'b', label = 'Discrete CMAC')
plt.plot(np.linspace(1,34,34),t_cont,'r', label = 'Continuous CMAC')
plt.legend(loc="upper left")
plt.title('Time to Convergence vs Overlap')
plt.xlabel('Overlap')
plt.ylabel('Time (s)')
plt.savefig('Time to Convergence vs Overlap.png')
plt.show()





