"""Set up data"""

from google.colab import drive
drive.mount('/content/drive')

import os
path = # Your path here
os.chdir(path)

#importing file with clean ICD data
import pandas as pd
data=pd.read_csv('final.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Features : ICD Codes (binary)
X = data[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'Z']]

# Target : Poverty status (binary)
y = data['POV']

# Convert to np arrays 
X = np.array(X)
y = np.array(y)

# Train/test split 
# We will only use X_train and y_train for training 
# X_test and y_test will only be used at the end for testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""Design preliminary NN architecture (no cross validation)"""

# Set hyperparameters 
opt_adam1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
opt_adam2 = tf.keras.optimizers.Adam(learning_rate=0.001)

m1 = Sequential([
    layers.Dense(12, activation='relu', input_shape=(20,)), # 12 neurons in the 1st hidden layer 
    layers.Dense(1,  activation='sigmoid')                  # Binary outcome 
])

m2 = Sequential([
    layers.Dense(12, activation='relu', input_shape=(20,)), # 12 neurons in 1st hidden layer 
    layers.Dense(9,  activation='relu'),                    #  9 neurons in 2nd hidden layer 
    layers.Dense(1,  activation='sigmoid')                  #  Binary outcome 
])

m3 = Sequential([
    layers.Dense(12, activation='relu', input_shape=(20,)), # 12 neurons in the 1st hidden layer
    layers.Dense(9,  activation='relu'),                    #  9 neurons in the 2nd hidden layer 
    layers.Dense(6,  activation='relu'),                    #  6 neurons in the 3rd hidden layer 
    layers.Dense(1,  activation='sigmoid')                  #  Binary outcome
])

m4 = Sequential([
    layers.Dense(12, activation='relu', input_shape=(20,)), # 12 neurons in the 1st hidden layer  
    layers.Dense(9,  activation='relu'),                    #  9 neurons in the 2nd hidden layer 
    layers.Dense(6,  activation='relu'),                    #  6 neurons in the 3rd hidden layer
    layers.Dense(3,  activation='relu'),                    #  3 neurons in the 4th hidden layer
    layers.Dense(1,  activation='sigmoid')                  #  Binary outcome 
])

"""Compile preliminary models """

# Compiles with the Adam optimizer specified above 
# Uses the cross entropy loss function and accuracy as metric 

m1.compile(optimizer=opt_adam1, loss=BinaryCrossentropy(),
           metrics=['accuracy'])

m2.compile(optimizer=opt_adam1, loss=BinaryCrossentropy(), 
           metrics=['accuracy'])

m3.compile(optimizer=opt_adam1, loss=BinaryCrossentropy(), 
           metrics=['accuracy'])

m4.compile(optimizer=opt_adam1, loss=BinaryCrossentropy(), 
           metrics=['accuracy'])

"""Define function to plot loss convergence"""

# Plots loss convergence from training process 

def plot_loss(history, title):
    plt.figure(figsize=(6, 4))
    plt.title('Loss Convergence for ' + title)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

"""Fit models and obtain preliminary results"""

fit1 = m1.fit(X_train, y_train, epochs=30, 
              verbose=0, validation_split=0.3)

plot_loss(fit1, 'Model 1 : [12]')

m1_loss, m1_acc = m1.evaluate(X_test, y_test, verbose=2)

fit2 = m2.fit(X_train, y_train, epochs=30, 
              verbose=0, validation_split=0.3)

plot_loss(fit2, 'Model 2 : [12, 9]')

m2_loss, m2_acc = m2.evaluate(X_test, y_test, verbose=2)

fit3 = m3.fit(X_train, y_train, epochs=30, 
              verbose=0, validation_split=0.3)

plot_loss(fit3, 'Model 3 : [12, 9, 6]')

m3_loss, m3_acc = m3.evaluate(X_test, y_test, verbose=2)

fit4 = m4.fit(X_train, y_train, epochs=30, 
              verbose=0, validation_split=0.3)

plot_loss(fit4, 'Model 4 : [12, 9, 6, 3]')

m4_loss, m4_acc = m4.evaluate(X_test, y_test, verbose=2)

"""Hyperparemeter tuning (Cross validation for grid search)"""

# Candidate learning rates 
adam_alpha = np.linspace(0.0001, 0.001, 10)

# Candidate L1 coefficients 
regl_coeff = np.linspace(0.0001, 0.05, 10)

# mspec   : number of neurons in each layer from 1st hidden layer to last hidden layer 
# ex) [12, 9, 6] -> 12 in 1st hidden layer, 9 in 2nd hidden layer, etc. 

# adam    : list of candidate learning rates for Adam optimizer
# regl    : list of candidate L1 coefficients 
# returns : compiled feed forward neural network model 

def build_model(mspec, adam, regl):
  
    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    model = Sequential() # Feed forward neural network 
    
    for m in mspec: # For each layer in model specification 
        
        if regl == 0: # If candidate L1 coefficients not specified/included
            model.add(layers.Dense(m, activation='relu')) # Layer with no regularizer 
            
        else: # If candidate L1 coefficients are specified/included 
            model.add(layers.Dense(m, activation='relu', kernel_regularizer=l1(regl))) # Layer with L1 norm 
            
    model.add(layers.Dense(1, activation='sigmoid')) # Final output layer
    
    opt_adam = tf.keras.optimizers.Adam(learning_rate=adam) # Adam optimizer with specified learning rate 
    
    model.compile(optimizer=opt_adam, loss=BinaryCrossentropy(), 
                  metrics=['accuracy']) # Compile model 
                  
    return model

# mspec   : model specification 
# adam    : list of candidate learning rates for Adam optimizer
# X, y    : training data as numpy arrays 
# returns : mean accuracy across folds for each learning rate 

def grid_search(mspec, adam, X, y): 
  
    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    adam_values = [] # Stores current learning rate
    accuracies  = [] # Stores current mean accuracy 
    
    kf = KFold(n_splits=5, random_state=0, shuffle=True) # 5-fold split 
    
    for a in adam: # For each candidate learning rate
        
        model = build_model(mspec, a, 0) # Build model with specified learning rate 
        fold_acc = [] # Stores accuracies from each fold 
        
        for train_idx, test_idx in kf.split(X, y): # Fit model for each fold 
            model.fit(X[train_idx], y[train_idx],
                      epochs=5, batch_size=512, verbose=0)
            
            loss, acc = model.evaluate(X[test_idx], y[test_idx], verbose=0) # Accuracy from each fold 
            fold_acc.append(acc) # Accuracy from each fold  
            
        mean_acc = np.mean(fold_acc) # Get mean accuracy from all 5 folds 
        
        adam_values.append(a) # Append current learning rate 
        accuracies.append(mean_acc) # Append current mean accuracy 
            
    return pd.DataFrame({'Learning Rate' : adam_values,
                         'Accuracy'      : accuracies})

# mspec   : model specification
# adam    : list of candidate learning rates for Adam optimizer
# regl    : list of candidate L1 coefficients 
# X, y    : training data as numpy arrays 
# returns : mean accuracy across folds for each combination of 
#           learning rates and L1 coefficients 

def grid_search_regl(mspec, adam, regl, X, y):

    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    adam_values = [] # Stores current learning rate
    regl_values = [] # Stores current L1 coefficient
    accuracies  = [] # Stores current mean accuracy
    
    kf = KFold(n_splits=5, random_state=0, shuffle=True) # 5-fold split 
    
    grid = list(itertools.product(adam, regl)) # All combinations of L1 coefficients and learning rates
    
    for g in grid: # Same process as grid_search() function above 
        
        model = build_model(mspec, g[0], g[1]) # Build/compile model with specified hyperparameters
        fold_acc = [] 
        
        for train_idx, test_idx in kf.split(X, y): # Fit model for each fold 
            model.fit(X[train_idx], y[train_idx],
                      epochs=5, batch_size=512, verbose=0)
            
            loss, acc = model.evaluate(X[test_idx], y[test_idx], verbose=0) # Accuracy from each fold 
            fold_acc.append(acc) # Append ccuracy from each fold 
            
        mean_acc = np.mean(fold_acc) # Get mean accuracy from all 5 folds 
        
        adam_values.append(g[0]) # Append current learning rate 
        regl_values.append(g[1]) # Append current L1 coefficient 
        accuracies.append(mean_acc) # Append current mean accuracy  
            
    return pd.DataFrame({'Learning Rate'  : adam_values, 
                         'L1 Coefficient' : regl_values, 
                         'Accuracy'       : accuracies})

# Plots heat map of grid search results with regularization 

def get_heat_map(heat_data, title):
    heat_data['Learning Rate'] = heat_data['Learning Rate'].round(4)
    heat_data['L1 Coefficient'] = heat_data['L1 Coefficient'].round(6) 
    heat_data_pivot = heat_data.pivot('Learning Rate', 'L1 Coefficient', 'Accuracy') 
    heat_map = sns.heatmap(heat_data_pivot).set_title('Mean Accuracies for ' + title)
    
    return heat_map

# Plots results of grid search without regularization 

def get_plot(data, title):
    plt.plot(data['Learning Rate'], data['Accuracy'])
    plt.title(title)
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Accuracy')

m1_history = grid_search([12], adam_alpha, X_train, y_train)
m1_history_regl = grid_search_regl([12], adam_alpha, regl_coeff, X_train, y_train)

get_plot(m1_history, 'Model 1 : [12]')

fig1 = get_heat_map(m1_history_regl, 'Model 1 : [12]')

m2_history = grid_search([12, 9], adam_alpha, X_train, y_train)
m2_history_regl = grid_search_regl([12, 9], adam_alpha, regl_coeff, X_train, y_train)

get_plot(m2_history, 'Model 2 : [12, 9]')

fig2 = get_heat_map(m2_history_regl, 'Model 2 : [12, 9]')

m3_history = grid_search([12, 9, 6], adam_alpha, X_train, y_train)
m3_history_regl = grid_search_regl([12, 9, 6], adam_alpha, regl_coeff, X_train, y_train)

get_plot(m3_history, 'Model 3 : [12, 9, 6]')

fig3 = get_heat_map(m3_history_regl, 'Model 3 : [12, 9, 6]')

m4_history = grid_search([12, 9, 6, 3], adam_alpha, X_train, y_train)
m4_history_regl = grid_search_regl([12, 9, 6, 3], adam_alpha, regl_coeff, X_train, y_train)

get_plot(m4_history, 'Model 4 : [12, 9, 6, 3]')

fig4 = get_heat_map(m4_history_regl, 'Model 4 : [12, 9, 6, 3]')

# returns : accuracy on test (hold out) set using the optimal hyperparameters for each model 
#           found in grid search above 

def final_results(mspec, adam, regl, X_train, y_train, X_test, y_test):

    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    accuracies, predictions = [], [] # Stores accuracies and predictions of model 
    
    for i in range(len(mspec)): # For each model 
        model = Sequential() 
        opt_adam = tf.keras.optimizers.Adam(learning_rate=adam[i]) # Specified learning rate 
        
        for m in mspec[i]: # Add hidden layers to model based on model specification
            if regl == 0: # If regularization specified 
                model.add(layers.Dense(m, activation='relu'))

            else: # If regularization not specified 
                model.add(layers.Dense(m, activation='relu', 
                                       kernel_regularizer=l1(regl[i]))) # Specified L1 coefficient 
        
        model.add(layers.Dense(1, activation='sigmoid')) # Final output layer 
        
        model.compile(optimizer=opt_adam, loss=BinaryCrossentropy(),
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=30, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(acc) # Record test accuracy 
        
        pred = model.predict(X_test)
        predictions.append(pred) # Record predictions on test set (as probabilities)
        
    return accuracies, predictions

# Model specifications 
models = [[12], [12,9], [12,9,6], [12,9,6,3]]

# Indices of optimal learning rates 
idx_m1 = np.argmax(m1_history.Accuracy) 
idx_m2 = np.argmax(m2_history.Accuracy)
idx_m3 = np.argmax(m3_history.Accuracy)
idx_m4 = np.argmax(m4_history.Accuracy)

# List of optimal learning rates from model 1 to model 4
adams = [m1_history['Learning Rate'][idx_m1], 
         m2_history['Learning Rate'][idx_m2],
         m3_history['Learning Rate'][idx_m3],
         m4_history['Learning Rate'][idx_m4]]

# Print optimal learning rates from model 1 to model 4 
print(adams) 

# Get accuracies and predictions on test set from NN models with NO regularization 
a, p = final_results(models, adams, 0, X_train, y_train, X_test, y_test)

# Indices of optimal hyperparameters 

idx_m1 = np.argmax(m1_history_regl.Accuracy)
idx_m2 = np.argmax(m2_history_regl.Accuracy)
idx_m3 = np.argmax(m3_history_regl.Accuracy)
idx_m4 = np.argmax(m4_history_regl.Accuracy)

# Optimal learning rates from model 1 to model 4
adams = [m1_history_regl['Learning Rate'][idx_m1],
         m2_history_regl['Learning Rate'][idx_m2],
         m3_history_regl['Learning Rate'][idx_m3],
         m4_history_regl['Learning Rate'][idx_m4]]

# Optimal L1 coefficients from model 1 to model 4
regls = [m1_history_regl['L1 Coefficient'][idx_m1],
         m2_history_regl['L1 Coefficient'][idx_m2],
         m3_history_regl['L1 Coefficient'][idx_m3],
         m4_history_regl['L1 Coefficient'][idx_m4]]

print(adams) # Print optimal learning rates from model 1 to model 4 
print(regls) # Print optimal L1 coefficients from model 1 to model 4 

# Get accuracies and predictions on test set from NN models WITH regularization 
a_r, p_r = final_results(models, adams, regls, X_train, y_train, X_test, y_test)

# Summary of final results on test set 
summary = pd.DataFrame({'Model' : [[12], [12,9], [12,9,6], [12,9,6,3]], 
                        'Test Accuracy w/o L1 Regularization'  : a,
                        'Test Accuracy w/ L1 Regularization'   : a_r})

summary.index = summary['Model']
summary.drop('Model', axis=1)

# Model 1 : [12] w/o L1 Regularization is optimal model

# Confusion matrix of optimal NN model 
tf.math.confusion_matrix(labels=y_test, predictions=p[0], num_classes=2)

# FPR and TPR from optimal NN model 
fpr, tpr, thresholds = roc_curve(y_test, p[0])

# ROC curve from optimal NN model 
line45 = np.linspace(0, 1, 100)

plt.plot(fpr, tpr)
plt.plot(line45, line45, linestyle="--")
plt.title("ROC Curve for Optimal NN Model : [12]")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# AUC from optimal NN model 
auc(fpr, tpr)

# Code to obtain loss convergence from fitting the optimal NN model to the train set  

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

optimal_model = Sequential([
    layers.Dense(12, activation='relu', input_shape=(20,)), 
    layers.Dense(1,  activation='sigmoid')                 
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

optimal_model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), 
                      metrics=['accuracy'])

optimal_fit = optimal_model.fit(X_train, y_train, epochs=30, 
                                verbose=0, validation_split=0.3)

# Plot loss convergence from fitting the optimal NN model to the train set  
plot_loss(optimal_fit, 'Optimal NN Model : [12]')

# Print out of the grid search for the optimal NN model 
print(m1_history)