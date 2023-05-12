#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm

class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        #checking for null values
        self.raw_input.isna().sum()

        #correlation
        corr = self.raw_input.corr()
        #sns.heatmap(self.raw_input=corr, annot=True)
        corr['PE'].abs().sort_values(ascending = False)

        #removing colums with 
        self.raw_input=self.raw_input.drop(columns=['AP','RH'])
        sklearn.preprocessing.scale(self.raw_input.AT)
        sklearn.preprocessing.scale(self.raw_input.V)
        sklearn.preprocessing.scale(self.raw_input.PE)
        self.processed_data = self.raw_input
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        
        y=np.asarray(y)
      #  print(y)
        #print(y.shape)
       
        # Below are the hyperparameters that you need to use for model
        #   evaluation
   
        model_eval_test=[]

        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        count=0
        color = cm.rainbow(np.linspace(0, 1, 24))

        for act_func in activations:
            for lr in learning_rate:
                for hid_ly in num_hidden_layers:
                    temp_x=[]
                    temp_y=[]
                    for itr in max_iterations:
                        count+=1
                        mlp=MLPClassifier(hidden_layer_sizes=hid_ly,activation=act_func,learning_rate_init=lr,max_iter=itr)
                        mlp.fit(X_train,y_train)
                        pred_train=mlp.predict(X_train)
                        acc_train=sklearn.metrics.accuracy_score(y_train,pred_train)
                        err_train=sklearn.metrics.mean_squared_error(y_train,pred_train)

                        pred_test=mlp.predict(X_test)
                        acc_test=sklearn.metrics.accuracy_score(y_test,pred_test)
                        err_test=sklearn.metrics.mean_squared_error(y_test,pred_test)

                        model_eval_test.append([act_func,lr,hid_ly,itr,acc_test,err_test])
                       #model_eval_train.append([acc_test,err_test])
                       #print("Model details- Activation function="+str(act_func)+" Learning rate="+str(lr)+" Epochs="+str(itr)+" Number of hidden layers="+str(hid_ly)+"--> train accuracy="+str(acc_train)+"train error="+str(err_train)+"test accuracy="+str(acc_test)+" test error="+str(err_test))
                        temp_x.append(itr)
                        temp_y.append(acc_test)
                    lb="Model "+str(count/2)
                    plt.plot(temp_x,temp_y,label=lb,c=color[count-1])

        plt.legend(bbox_to_anchor = (1.10, 0.8))
        f =plt.figure(figsize=(500,300))
        plt.show()
        df = pd.DataFrame(model_eval_test, columns =['Activation function','Learning rate','Number of hidden layer','Epochs','Model Accuracy','Model Error'])
        print(df.sort_values(by='Model Accuracy',ascending=False))
        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/ojal21/ML_Dataset/main/CCPP.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
