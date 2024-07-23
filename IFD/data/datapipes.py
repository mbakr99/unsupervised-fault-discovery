import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from math import floor
from typing import Iterable,Any


#This class is based on Tensorflow tutorial (Time series prediction)
class Window:
  def __init__(self,input_width,label_width,shift,tr_data: pd.DataFrame,val_data: pd.DataFrame,test_data: pd.DataFrame,batch_size: int =264,target_names: Iterable[str]=None) -> None:

    self.train_df=tr_data
    self.validation_df=val_data
    self.test_df=test_data
    self.batch_size=batch_size

    self._example=None

    self.input_width=input_width
    self.label_width=label_width
    self.shift=shift #TODO: For the case of using multiple samples this part might cause a problem. Because the sequence width will be= in_width+out_width
    # (However, as long as the shift is set to 1 and the output width is also 1 no problem should be encountered)
    self.tot_width=input_width+shift

    self.input_slice=slice(0,input_width) #this is needed in indexing tensors (EagerTensor)
    self.input_indices=np.arange(self.tot_width)[self.input_slice]
    self.label_slice=slice(self.tot_width-label_width,None) #self.tot_width-label_width this is the starting index of the label
    self.label_indices=np.arange(self.tot_width)[self.label_slice]

    self.target_names=target_names
    self.features_names_dic={name:i for i,name in enumerate(tr_data.columns)} #this assumes that tr_data is in the form of a dataframe. It is a dictionary saving the names of the features along with their index
    if target_names is not None:
      self.target_names_dic={name:i for i,name in enumerate(target_names)} #target_names is expected to be an iterable  (I indicated this using type hints)

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.tot_width}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.features_names_dic}'])

  def split(self,data): #data has the structure of [batches,time,features]
    #pdb.set_trace()
    input_data=data[:,self.input_slice,:] #<TODO> this also means all the features are considered in the input vector. Modify this issue for a better and more general usage
    output_data=data[:,self.label_slice,:]
    if self.target_names is not None:
      output_data= tf.stack(
        [output_data[:, :, self.features_names_dic[name]] for name in self.target_names],
        axis=-1) #stacking is done along the innermost axis (features axis)

    # Slicing doesn't preserve static shape information, so set the shapes    (tensorflow )
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    #I don't understand why the shape will not be preserved after slicing #I checked this on a dummy variable, when using the tf.stack an extra dimension is added. Thus, setting the shape manually ensures that the static shape info
    #is conserved (Me)
    input_data.set_shape([None, self.input_width, None])
    output_data.set_shape([None, self.label_width, None])
    return input_data,output_data

  def plot(self,model=None,plot_var : str=None,max_subplots=3):
    input,output=self.example #this will return a batch of df data [batch,time,features]
    plt.figure(figsize=(12, 8))
    plot_var_idx = self.features_names_dic[plot_var]
    max_n = min(max_subplots, len(input))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_var} [normed]')
      plt.plot(self.input_indices, input[n, :, plot_var_idx],
               label='Inputs', marker='.', zorder=-10)

      #check if the plot_var is part of the target variables (output)
      if self.target_names: #if this is not None
        plot_label_var_idx=self.target_names_dic.get(plot_var,None) #in this case the output features are a subset of the full features set
        #this is why the target_names_dic is used instead of the features_names_dic
      else: #in this case the output features are the same as the input features (full feature set)
        plot_label_var_idx=plot_var_idx

      if plot_label_var_idx is None: #the user selected a feature/variable that is not part of the output features
        continue

      plt.scatter(self.label_indices, output[n, :, plot_label_var_idx],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)

      if model is not None:
        predictions = model(input)
        plt.scatter(self.label_indices, predictions[n, :, plot_label_var_idx],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
      if n==0:
        plt.legend()

      plt.xlabel('Time Step') #this should be adjusted to reflect the time-step size of the process or the sampling time

  def create_dataset(self,data: pd.DataFrame):
    data=np.array(data,dtype='float32') #cast pd.DataFrame into a float32 numpy array
    ds=keras.utils.timeseries_dataset_from_array(data,targets=None,sequence_length=self.tot_width,
                                                 sequence_stride=1,sampling_rate=1,batch_size=self.batch_size)
    ds=ds.map(self.split).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

  @property
  def train_ds(self):
    return self.create_dataset(self.train_df)

  @property
  def val_ds(self):
    return self.create_dataset(self.validation_df)

  @property
  def test_ds(self):
    return self.create_dataset(self.test_df)

  @property
  def example(self):
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train_ds))
      # And cache it for next time
      self._example = result
    return result



def shuffle_split(data: pd.DataFrame,tr_frac: float,val_frac: float,time_stamp=None,shuffle_flag=False) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,Any]:
  if shuffle_flag:
    data=data.sample(frac=1)
  n=len(data)
  #get the start and end indices of the training, validation, and testing sets
  tr_end=floor(tr_frac*n)
  val_start=tr_end
  val_end=tr_end+floor(val_frac*n)
  test_start=val_end
  #split the data
  tr_df=data.iloc[:tr_end,:]
  val_df=data.iloc[val_start:val_end,:]
  test_df=data.iloc[test_start:,:]
  print("The length of the training set is {} \nThe length of the validation set is {} \nThe length of the testing set is {}".format(len(tr_df),len(val_df),len(test_df)))
  return tr_df,val_df,test_df, time_stamp




class Window_Var(Window):
  
  def __init__(self,input_width,label_width,shift,tr_data: pd.DataFrame,val_data: pd.DataFrame,test_data: pd.DataFrame,batch_size: int =264,target_names: Iterable[str]=None):
    super(Window_Var,self).__init__(input_width=input_width,label_width=label_width,shift=shift,tr_data=tr_data,val_data=val_data,test_data=test_data,batch_size=batch_size,target_names=target_names)

  def plot(self,model=None,plot_var: str=None,max_subplots=3):
    input,output=self.example #this will return a batch of df data [batch,time,features]
    plt.figure(figsize=(12, 8))
    plot_var_idx = self.features_names_dic[plot_var]
    max_n = min(max_subplots, len(input))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_var} [normed]')
      plt.plot(self.input_indices, input[n, :, plot_var_idx],
               label='Inputs', marker='.', zorder=-10)

      #check if the plot_var is part of the target variables (output)
      if self.target_names: #if this is not None
        plot_label_var_idx=self.target_names_dic.get(plot_var,None) #in this case the output features are a subset of the full features set
        #this is why the target_names_dic is used instead of the features_names_dic
      else: #in this case the output features are the same as the input features (full feature set)
        plot_label_var_idx=plot_var_idx

      if plot_label_var_idx is None: #the user selected a feature/variable that is not part of the output features
        continue

      plt.scatter(self.label_indices, output[n, :, plot_label_var_idx],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)

      if model is not None:
        predictions = model(input)
        predictions=predictions[0] #get the mean of the distribution
        plt.scatter(self.label_indices, predictions[n, :, plot_label_var_idx],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
      if n==0:
        plt.legend()

      plt.xlabel('Time Step') #this should be adjusted to reflect the time-step size of the process or the sampling time


def normalize_data(data:pd.DataFrame) ->pd.DataFrame:
  data_m=data.mean()
  data_std=data.std()
  return (data-data_m)/data_std
