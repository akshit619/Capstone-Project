import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.restoration import denoise_wavelet
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import time
import math



class Util:
    @staticmethod
    @staticmethod
    def plot_subplots(data):
      fig = plt.figure(figsize=(20, 12))
      data.plot(subplots=True, figsize=(12, 10), grid=False)
      sns.set_style("whitegrid")
      plt.show()

    @staticmethod
    @staticmethod
    def denoise_data(data):
        # Assuming the first column contains the 'Close' data
        close_data = data.iloc[:, 0]
        denoised_close = denoise_wavelet(close_data, wavelet='haar', method='VisuShrink', mode='soft', rescale_sigma=True)
        data['Close'] = denoised_close
        return data

    @staticmethod
    @staticmethod
    def plot_corr_heatmap(data):
        plt.figure(figsize=(10, 5))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap')
        plt.show()

    @staticmethod
    @staticmethod
    def plot_ma_trend(data):
        df = data.copy()
        df['Moving_Avg_50'] = df['Close'].rolling(50).mean()
        df['Moving_Avg_200'] = df['Close'].rolling(200).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(df['Close'], color='tab:blue', label='NIFTY 50 Close Price', linewidth=2)
        plt.plot(df['Moving_Avg_50'], color='tab:red', label='50-day MA', linewidth=2)
        plt.plot(df['Moving_Avg_200'], color='tab:green', label='200-day MA', linewidth=2)
        plt.legend(loc='upper left')
        plt.title('Moving Averages Trend')
        plt.xlabel('Time (years)')
        plt.ylabel('Close price')
        plt.grid(True)
        plt.show()
  
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100

    
    @staticmethod
    def calculate_scores(y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        R = np.corrcoef(y_true, y_pred)[0, 1]
        mape = Util.mean_absolute_percentage_error(y_true, y_pred)
        scores = {'rmse': rmse, 'R': R, 'mape': mape}
        return scores

    
    @staticmethod
    def create_dataset(dataset, time_step=1):
        DataX, DataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), ]
            DataX.append(a)
            DataY.append(dataset[i + time_step, 0])  # ydata consists close price
        return np.array(DataX), np.array(DataY)

    
    @staticmethod
    def data_split(data, split=0.2):
        l1 = int(len(data) * (1 - split))
        data1 = data.iloc[0:l1, :]
        data2 = data.iloc[l1:len(data), :]
        return data1, data2

    
    @staticmethod
    def min_max_transform(data, feature_range=(0, 1)):
        scaler = MinMaxScaler(feature_range)
        return scaler.fit_transform(data)

    
    @staticmethod
    def min_max_inverse_transform(data_scaled, min_original, max_original):
        return min_original + data_scaled * (max_original - min_original)

    
    @staticmethod
    def write_dict_to_file(dic_name, file_name):
        with open(file_name, 'w') as file:
            file.write(str(dic_name))

    
    @staticmethod
    def read_dict_from_file(file_name):
        with open(file_name, "r") as file:
            contents = file.read()
            dictionary = eval(contents)
        return dictionary
    @staticmethod
    def test_scores_plot(model_output):
      neurons = model_output['avg_scores']['neurons']
      rmse = model_output['avg_scores']['rmse']
      #mae =  model_output['avg_scores']['mae']
      mape =  model_output['avg_scores']['mape']
      #R2 =   model_output['avg_scores']['R2']
      R =    model_output['avg_scores']['R']
      #time =  model_output['avg_scores']['elapsed_time']

      fig = plt.figure(figsize = (18, 4))
      plt.subplot(131)
      plt.plot(neurons, rmse, '--o', linewidth = 2, color = 'indigo')
      plt.title("(a)")
      plt.xlabel("Neurons")
      plt.ylabel("Avg. RMSE")
      sns.set_style("whitegrid")


      plt.subplot(132)
      plt.plot(neurons, mape, '--o', linewidth = 2, color = 'darkgreen')
      plt.title("(b)")
      plt.xlabel("Neurons")
      plt.ylabel("Avg. MAPE")


      plt.subplot(133)
      plt.plot(neurons, R, '--o', linewidth = 2, color = 'darkred')
      plt.title("(c)")
      plt.xlabel("Neurons")
      plt.ylabel("Avg. R ")

      fig.savefig("res/multiple_avg_scores_plots.png",dpi=600)
      plt.show()


    @staticmethod
    def true_pred_plot(model_output):

      y_train = model_output['datasets']['y_train']
      y_test =  model_output['datasets']['y_test']

      train_pred = model_output['best_model']['train_predictions']
      test_pred = model_output['best_model']['test_predictions']

      ##====== Visualizing true vs predicted plots ========#
      fig = plt.figure(figsize= (14,5))
      plt.subplot(121)
      #sns.relplot(x = y_train_original, y = train_pred_original)
      plt.scatter(y_train, train_pred, marker= "+", color = 'mediumblue')
      identity_line = np.linspace(max(min(y_train), min(train_pred)), min(max(y_train), max(train_pred)))
      plt.plot(identity_line, identity_line, color="red", linestyle="dashed", linewidth= 2.5)

      plt.xlabel("True")
      plt.ylabel("Predicted")
      plt.title("(a)")

      plt.subplot(122)
      #sns.relplot(x = y_test_original, y = test_pred_original)
      plt.scatter(y_test, test_pred, marker = "+", color = 'mediumblue')
      identity_line = np.linspace(max(min(y_test), min(test_pred)), min(max(y_test), max(test_pred)))
      plt.plot(identity_line, identity_line, color="red", linestyle="dashed", linewidth= 2.5)
      plt.xlabel("True")
      plt.ylabel("Predicted")
      plt.title("(b)")
      fig.savefig("res/True_vs_predicted_plot.png", dpi=600)
      plt.show()


    @staticmethod
    def prediction_plot(model_output):
      time_step =  model_output['hyper_parameters']['time_step']
      best_replicate = model_output['best_model']['replicate']

      data = model_output['datasets']['data']
      print(data)
      data.to_csv("xyz.csv")

      train_predict_plot_data = np.empty_like(data.values[:,0])# extracting closing price
      train_predict_plot_data[:] = np.nan

      test_predict_plot_data = np.empty_like(data.values[:,0])
      test_predict_plot_data[:] = np.nan

      fig1 = plt.figure(figsize = (18,12))

      plt.subplot(231)

      train_pred = model_output['train_predictions'][0][best_replicate]
      test_pred = model_output['test_predictions'][0][best_replicate]


      train_predict_plot_data[time_step:len(train_pred)+ time_step] =  train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] = test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(a)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')

      plt.subplot(232)

      train_pred = model_output['train_predictions'][1][best_replicate]
      test_pred = model_output['test_predictions'][1][best_replicate]

      train_predict_plot_data[time_step:len(train_pred)+ time_step] =  train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] =  test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(b)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')

      plt.subplot(233)

      train_pred = model_output['train_predictions'][2][best_replicate]
      test_pred = model_output['test_predictions'][2][best_replicate]

      train_predict_plot_data[time_step:len(train_pred)+ time_step] = train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] =  test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(c)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')


      plt.subplot(234)

      train_pred = model_output['train_predictions'][3][best_replicate]
      test_pred = model_output['test_predictions'][3][best_replicate]

      train_predict_plot_data[time_step:len(train_pred)+ time_step] = train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] = test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(d)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')


      plt.subplot(235)

      train_pred = model_output['train_predictions'][4][best_replicate]
      test_pred = model_output['test_predictions'][4][best_replicate]

      train_predict_plot_data[time_step:len(train_pred)+ time_step] = train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] = test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(e)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')

      plt.subplot(236)

      train_pred = model_output['train_predictions'][5][best_replicate]
      test_pred = model_output['test_predictions'][5][best_replicate]

      train_predict_plot_data[time_step:len(train_pred)+ time_step] = train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] = test_pred

      plt.plot(data.values[:,0],'k',linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(f)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')


      fig1.savefig("res/predictions_plots_fullset.png",dpi=600)
      plt.show()

      fig2 = plt.figure(figsize = (18,12))

      plt.subplot(231)
      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][0][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(a)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')


      plt.subplot(232)

      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][1][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(b)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')


      plt.subplot(233)

      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][2][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(c)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')



      plt.subplot(234)

      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][3][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(d)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')

      plt.subplot(235)

      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][4][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(e)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')


      plt.subplot(236)

      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'mediumblue',linewidth = 1.5)
      plt.plot(model_output['test_predictions'][5][best_replicate], 'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(f)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')



      fig2.savefig("res/predictions_plots_testset.png",dpi=600)

      plt.show()

    @staticmethod
    def best_model_prediction_plot(model_output):

      time_step =  model_output['hyper_parameters']['time_step']

      data = model_output['datasets']['data']

      train_predict_plot_data = np.empty_like(data.values[:,0])# extracting closing price
      train_predict_plot_data[:] = np.nan

      test_predict_plot_data = np.empty_like(data.values[:,0])
      test_predict_plot_data[:] = np.nan


      fig = plt.figure(figsize = (14,5))

      plt.subplot(121)

      train_pred = model_output['best_model']['train_predictions']
      test_pred = model_output['best_model']['test_predictions']

      train_predict_plot_data[time_step:len(train_pred)+ time_step] =  train_pred
      test_predict_plot_data[len(train_pred)+(time_step*2)+1:len(data.values)-1] = test_pred

      plt.plot(data.values[:,0],'k', linewidth = 1.5)
      plt.plot(train_predict_plot_data,'mediumblue',linewidth = 1.5)
      plt.plot(test_predict_plot_data,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(a)")
      plt.legend(['True value', 'Predicted value in train set', 'Predicted value in test set'], loc = 'upper left')


      plt.subplot(122)
      plt.plot(data.values[len(train_pred)+(time_step*2)+1:-1, 0],'k',linewidth = 1.5)
      plt.plot(test_pred,'darkgreen',linewidth = 1.5)
      plt.xlabel('')
      plt.ylabel('Close price')
      plt.title("(b)")
      plt.legend(['True value', 'Predicted value'], loc='upper left')

      fig.savefig("res/best_model_predictions_plots.png",dpi=600)


    @staticmethod
    def rmse_boxplots(model_output):
      fig = plt.figure(figsize = (6,5))
      plt.boxplot(model_output['scores']['rmse'], patch_artist=True)
      plt.xticks([1,2,3,4,5,6], ['10', '30', '50', '100', '150', '200'])
      plt.xlabel('Number of neurons')
      plt.ylabel('RMSE')
      fig.savefig("res/rmse_boxplots.png",dpi=600)
      plt.show()


    @staticmethod
    def rmse_violinplotplots(model_output):
      fig = plt.figure(figsize = (6,5))
      plt.violinplot(model_output['scores']['rmse'])
      plt.xticks([1,2,3,4,5,6], ['10', '30', '50', '100', '150', '200'])
      plt.xlabel('Number of neurons')
      plt.ylabel('RMSE')
      fig.savefig("res/rmse_violinplots.png",dpi=600)
      plt.show()

    @staticmethod
    def all_scores_boxplots(model_output):

      fig = plt.figure(figsize = (18,5))
      plt.subplot(131)
      p1 = plt.boxplot(model_output['scores']['rmse'],patch_artist=True)
      for i, box in enumerate(p1['boxes']):
        # change outline color
        box.set(color= 'blue', linewidth = 1.2)
        # change fill color
        box.set(facecolor = 'mediumblue')
      plt.xticks([1,2,3,4,5,6], ['10', '30', '50', '100', '150', '200'])
      plt.title("(a)")
      plt.xlabel('Number of neurons')
      plt.ylabel('RMSE')

      plt.subplot(132)
      p2 = plt.boxplot(model_output['scores']['mape'],patch_artist=True)
      for i, box in enumerate(p2['boxes']):
        #change outline color
        box.set(color= 'blue', linewidth = 1.2)
        # change fill color
        box.set(facecolor = 'indigo')

      plt.xticks([1,2,3,4,5,6], ['10', '30', '50', '100', '150', '200'])
      plt.title("(b)")
      plt.xlabel('Number of neurons')
      plt.ylabel('MAPE')

      plt.subplot(133)
      p3 = plt.boxplot(model_output['scores']['R'],patch_artist=True)
      for i, box in enumerate(p3['boxes']):
        #change outline color
        box.set(color= 'blue', linewidth = 1.2)
        # change fill color
        box.set(facecolor = 'darkgreen')
      plt.xticks([1,2,3,4,5,6], ['10', '30', '50', '100', '150', '200'])
      plt.title("(c)")
      plt.xlabel('Number of neurons')
      plt.ylabel('R')

      fig.savefig("res/all_scores_boxplots.png",dpi=600)
      plt.show()


    @staticmethod
    def create_visualization(model_output):
        Util.true_pred_plot(model_output)  # Call true_pred_plot using Util class
        Util.test_scores_plot(model_output)
        Util.prediction_plot(model_output)
        Util.best_model_prediction_plot(model_output)
        Util.rmse_boxplots(model_output)
        Util.all_scores_boxplots(model_output)

    @staticmethod
    def build_model(layers, time_step, num_features, optimizer = 'Adam', learning_rate = 0.001, verbose = 1,drop =0.2):

      model = Sequential()
      for i in range(len(layers)):
        if len(layers)==1:
          model.add(LSTM(int(layers[i]), input_shape = (time_step, num_features)))
        else:
          if i < len(layers)-1:
            if i == 0:
              model.add(LSTM(int(layers[i]), input_shape=(time_step, num_features), return_sequences= True))
              model.add(Dropout(drop))
            else:
              model.add(LSTM(int(layers[i]), return_sequences=True))
              model.add(Dropout(drop))
          else:
            model.add(LSTM(int(layers[i])))
            model.add(Dropout(drop))
      model.add(Dense(1, activation = 'linear'))

      if optimizer == 'Adam':
        opt = optimizers.Adam(learning_rate = learning_rate)
      elif optimizer == 'Adagrad':
        opt = optimizers.Adagrad(learning_rate = learning_rate)
      elif optimizer == 'Nadam':
        opt = optimizers.Nadam(learning_rate = learning_rate)
      elif optimizer == 'Adadelta':
        opt = optimizers.Adadelta(learning_rate= learning_rate)
      elif optimizer == 'RMSprop':
        opt = optimizers.RMSprop(learning_rate= learning_rate)
      else:
        print("No optimizer found in the list(['Adam', 'Adagrad','Nadam', 'Adadelta', 'RMSprop'])! Please apply your optimizer manually...")

      model.compile(loss='mean_squared_error', optimizer= opt)

      if verbose == 1:
        print(model.summary())
      return model
    @staticmethod
    def create_output_dict(neuron_units, min_index, min_col, best_rmse, best_mape, best_R, best_elapsed_time,
                       train_predictions_best_rmse, test_predictions_best_rmse, loss_best_rmse,
                       hp_parameters, epochs, time_steps, num_runs, test_ratio,
                       rmse_df, mape_df, R_df, elapsed_time_df,
                       train_predictions, test_predictions, models_history,
                       dataset, X_train, X_test, y_train_orig, y_test_orig):

      #======= Collecting hyperparameters=============#
      hyper_parameters = { 'neurons': neuron_units,
                          'model_specific_hyper_parameters': hp_parameters,
                          'epochs': epochs,
                          'time_step': time_steps,
                          'num_replicates': num_runs,
                          'test_split': test_ratio

                        }

      #======= Collecting test scores =============#
      scores = {'neurons': pd.DataFrame(neuron_units),
                'rmse': rmse_df,
                'mape': mape_df,
                'R': R_df,
                'elapsed_time': elapsed_time_df}

      #======= Collecting average test scores =============#
      avg_scores = pd.DataFrame({'neurons': neuron_units,
                                'rmse': rmse_df.mean(axis=1),
                                'mape': mape_df.mean(axis=1),
                                'R': R_df.mean(axis=1),
                                'elapsed_time': elapsed_time_df.mean(axis=1)})

      #======= Collecting standard deviations =============#
      all_stds = pd.DataFrame({'neurons': neuron_units,
                                'rmse': rmse_df.std(axis=1),
                                'mape': mape_df.std(axis=1),
                                'R': R_df.std(axis=1),
                                'elapsed_time': elapsed_time_df.std(axis=1)})

      #======= Collecting minimums =============#
      all_minimums = pd.DataFrame({'neurons': neuron_units,
                                    'rmse': rmse_df.min(axis=1),
                                    'mape': mape_df.min(axis=1),
                                    'R': R_df.min(axis=1),
                                    'elapsed_time': elapsed_time_df.min(axis=1)})

      #======= Collecting maximums =============#
      all_maximums = pd.DataFrame({'neurons': neuron_units,
                                    'rmse': rmse_df.max(axis=1),
                                    'mape': mape_df.max(axis=1),
                                    'R': R_df.max(axis=1),
                                    'elapsed_time': elapsed_time_df.max(axis=1)})

      #======= Collecting the best model results =============#
      best_model_results = {'neurons': neuron_units[min_index],
                            'replicate': min_col,
                            'rmse': best_rmse,
                            'mape': best_mape,
                            'R':  best_R,
                            'elapsed_time': best_elapsed_time,
                            'train_predictions': train_predictions_best_rmse,
                            'test_predictions': test_predictions_best_rmse,
                            'loss': loss_best_rmse
                            }

      datasets = {'data': dataset,
                  'X_train': X_train,
                  'X_test': X_test,
                  'y_train': y_train_orig,
                  'y_test': y_test_orig
                  }

      #======= Collecting all the outputs together =============#
      output_dict = {'hyper_parameters': hyper_parameters,
                    'best_model': best_model_results,
                    'scores': scores,
                    'avg_scores': avg_scores,
                    'all_stds': all_stds,
                    'all_minimums': all_minimums,
                    'all_maximums': all_maximums,
                    'train_predictions': train_predictions,
                    'test_predictions': test_predictions,
                    'models_history': models_history,
                    'datasets': datasets
                    }

      return output_dict ,avg_scores , all_stds ,all_minimums,all_maximums 
    

    @staticmethod
    def build_multi_layer_LSTM(layers_config, hp_parameters, dataset, time_step=5, test_split=0.2, epochs=5, num_runs=2 , drop =  0.2 ):
      print("Progress: Performing data preparation steps.......\n")

      # Splitting data into training and test sets
      train_data, test_data = Util.data_split(dataset, test_split)

      num_features = train_data.shape[1]

      min_train, max_train = train_data["Close"].min(), train_data["Close"].max()
      min_test, max_test = test_data["Close"].min(), test_data["Close"].max()

      train_data_scaled = Util.min_max_transform(train_data)
      test_data_scaled = Util.min_max_transform(test_data)

      X_train, y_train = Util.create_dataset(train_data_scaled, time_step)
      X_test, y_test = Util.create_dataset(test_data_scaled, time_step)

      train_y_original = Util.min_max_inverse_transform(y_train, min_train, max_train)
      test_y_original = Util.min_max_inverse_transform(y_test, min_test, max_test)

      # Arrays for collecting test scores
      rmse_array = np.zeros(num_runs)
      mape_array = np.zeros(num_runs)
      R_array = np.zeros(num_runs)
      elapsed_time_array = np.zeros(num_runs)

      models_history = []
      train_predictions = []
      test_predictions = []

      for i in range(num_runs):
          print("Program is running for %d replicate ----->\n" % i)

          model = Util.build_model(layers_config, time_step, num_features, optimizer=hp_parameters[0], learning_rate=hp_parameters[1], verbose=0 , drop = drop)
          callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

          start_time = time.time()
          history = model.fit(X_train, y_train, batch_size=hp_parameters[2], epochs=epochs, callbacks=[callback], verbose=1)
          end_time = time.time()
          elapsed_time = end_time - start_time

          models_history.append(history)

          # Making train and test predictions in original scales
          train_pred = Util.min_max_inverse_transform(model.predict(X_train).ravel(), min_train, max_train)
          test_pred = Util.min_max_inverse_transform(model.predict(X_test).ravel(), min_test, max_test)

          train_predictions.append(train_pred)
          test_predictions.append(test_pred)

          # Calculating performance scores
          scores = Util.calculate_scores(Util.min_max_inverse_transform(y_test, min_test, max_test), test_pred)

          rmse_array[i] = scores['rmse']
          mape_array[i] = scores['mape']
          R_array[i] = scores['R']
          elapsed_time_array[i] = elapsed_time

      min_index = rmse_array.argmin()
      best_rmse = rmse_array[min_index]
      mape_with_best_rmse = mape_array[min_index]
      R_with_best_rmse = R_array[min_index]
      elapsed_time_with_best_rmse = elapsed_time_array[min_index]

      train_predictions_with_best_rmse = train_predictions[min_index]
      test_predictions_with_best_rmse = test_predictions[min_index]

      loss_with_best_rmse = models_history[min_index].history['loss']

      output_dictionary = Util.create_output_dictionary(layers_config, hp_parameters, dataset, time_step, test_split, epochs, num_runs, num_features, min_train, max_train, min_test, max_test, train_y_original, test_y_original, rmse_array, mape_array, R_array, elapsed_time_array, train_predictions, test_predictions, models_history)
      

      return output_dictionary
    @staticmethod
    def create_output_dictionary(layers_config, hp_parameters, dataset, time_step, test_split, epochs, num_runs, num_features, min_train, max_train, min_test, max_test, train_y_original, test_y_original, rmse_array, mape_array, R_array, elapsed_time_array, train_predictions, test_predictions, models_history):
      min_index = rmse_array.argmin()
      best_rmse = rmse_array[min_index]
      mape_with_best_rmse = mape_array[min_index]
      R_with_best_rmse = R_array[min_index]
      elapsed_time_with_best_rmse = elapsed_time_array[min_index]

      train_predictions_with_best_rmse = train_predictions[min_index]
      test_predictions_with_best_rmse = test_predictions[min_index]

      loss_with_best_rmse = models_history[min_index].history['loss']

      # Collecting scores
      all_scores = {'rmse': rmse_array, 'mape': mape_array, 'R': R_array, 'elapsed_time': elapsed_time_array}

      # Collecting average scores
      avg_scores = {'rmse': np.mean(rmse_array),
                    'mape': np.mean(mape_array),
                    'R': np.mean(R_array),
                    'elapsed_time': np.mean(elapsed_time_array)}

      # Collecting standard deviations of scores
      stds = {'rmse': np.std(rmse_array),
              'mape': np.std(mape_array),
              'R': np.std(R_array),
              'elapsed_time': np.std(elapsed_time_array)}

      # Collecting minimum values of test scores
      minimums = {'rmse': np.min(rmse_array),
                  'mape': np.min(mape_array),
                  'R': np.min(R_array),
                  'elapsed_time': np.min(elapsed_time_array)}

      # Collecting maximum values of test scores
      maximums = {'rmse': np.max(rmse_array),
                  'mape': np.max(mape_array),
                  'R': np.max(R_array),
                  'elapsed_time': np.max(elapsed_time_array)}

      model_with_best_rmse = {
          'replicate': min_index,
          'rmse': best_rmse,
          'mape': mape_with_best_rmse,
          'R': R_with_best_rmse,
          'elapsed_time': elapsed_time_with_best_rmse,
          'train_predictions': train_predictions_with_best_rmse,
          'test_predictions': test_predictions_with_best_rmse,
          'loss': loss_with_best_rmse,
      }

      # Collecting hyperparameters
      hyper_parameters = {
          'layers_config': layers_config,
          'model_specific_hyper_parameters': hp_parameters,
          'epochs': epochs,
          'time_step': time_step,
          'num_runs': num_runs,
          'test_split': test_split
      }

      datasets = {
          'data': dataset,
          'train_y_original': train_y_original,
          'test_y_original': test_y_original
      }

      # Collecting all the outputs together
      output_dictionary = {
          'hyper_parameters': hyper_parameters,
          'best_model': model_with_best_rmse,
          'all_scores': all_scores,
          'avg_scores': avg_scores,
          'standard_deviations': stds,
          'minimums': minimums,
          'maximums': maximums,
          'train_predictions': train_predictions,
          'test_predictions': test_predictions,
          'datasets': datasets
      }

      return output_dictionary


