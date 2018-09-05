import os
#import math
from sklearn.model_selection import StratifiedKFold
import numpy as np
from FileDataGenerator import FileDataGen
#import copy
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)



class K_Fold: 
    def __init__(self, path, k):
        data=[]
        labels=[]
        for class_ in os.listdir(path):
            dat = [os.path.join(path, class_, img) for img in os.listdir(os.path.join(path, class_))]
            lab = [class_ for i in os.listdir(os.path.join(path, class_))]
            labels = labels+lab
            data = data + dat
        
        self.folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1234).split(data, labels))
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.k=k
        self.hist_dict = dict()
    
    def Apply_KFold(self, model, train_gen_params, test_gen_params, train_params, test_params, fit_params):
        
        self.hist_dict = dict()
        
        Wsave = model.get_weights()
        
        train_datagen = FileDataGen(**train_gen_params) 
        val_datagen = FileDataGen(**test_gen_params) 
        
        for j, (train_idx, val_idx) in enumerate(self.folds):
            print('\nFold ',j)
            X_train_cv = self.data[train_idx]
            y_train_cv = self.labels[train_idx]
            X_valid_cv = self.data[val_idx]
            y_valid_cv= self.labels[val_idx]
            
            model.set_weights(Wsave) #re-initialize weights 
            
            train_gen = train_datagen.flow_from_filelist(
                X_train_cv,
                y_train_cv,
                **train_params)
            
            val_gen = val_datagen.flow_from_filelist(
                X_valid_cv,
                y_valid_cv,
                **test_params)
            
            print('Training')
            hist=model.fit_generator(
                generator = train_gen,
                steps_per_epoch=len(X_train_cv)/train_params['batch_size'],
                **fit_params,
                validation_steps=len(X_valid_cv)/test_params['batch_size'],
                validation_data = val_gen)
            
            if len(self.hist_dict) == 0:
                #self.hist_dict = copy.deepcopy(hist.history) #Save all the data
                for key, val in hist.history.items(): #Just save at the end of the epoch
                    self.hist_dict[key]= [val[len(val)-1]]
            else:
                for key, val in hist.history.items(): #Just save at the end of the epoch
                    self.hist_dict[key].append(val[len(val)-1])
                    #for i in val:  #Save all the data
                        #self.hist_dict[key].append(i)
        return self.hist_dict
    
    def Check_Folds(self):
        print('There are {} Folds'.format(len(self.folds)))
        #print('Train data contains {} samples'.format(len(self.folds[0][0])))
        #print('Test data contains {} samples'.format(len(self.folds[0][1])))
        #print('Test data samples are aprox computed from ceil(len(data)/k): {}'.format(math.ceil(len(self.data)/self.k)))
        for j, (train_idx, val_idx) in enumerate(self.folds):
            print('\nFold ',j)
            X_train_cv = self.data[train_idx]
            y_train_cv = self.labels[train_idx]
            X_valid_cv = self.data[val_idx]
            y_valid_cv= self.labels[val_idx]
            unique, counts = np.unique(y_train_cv, return_counts=True)
            print('For training, {0} samples: {1}'.format(len(X_train_cv), dict(zip(unique, counts))))
    
            unique, counts = np.unique(y_valid_cv, return_counts=True)
            print('For testing, {0} samples: {1}'.format(len(X_valid_cv), dict(zip(unique, counts))))
    
            print('First five X_train images: \n{}'.format(X_train_cv[:5]))
            print('First five X_val images: \n{}'.format(X_valid_cv[:5]))