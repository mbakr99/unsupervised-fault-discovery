
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from typing import Iterable
from numpy import arange
import pdb


class TepDataPipe:

  def __init__(self, input_width: int, label_width: int, shift: int, tr_data: pd.DataFrame, val_data: pd.DataFrame,
               test_data: pd.DataFrame, target_names: Iterable[str], sampling_rate=1, seqeunce_stride=1,batch_size: int = 64,
               shuffle_flag: bool = False, flag_imglike_data: bool = False):
    """
    class for handling the visulization, tf.data.Dataset creation of the TEP data

    : param tr_data: pd.DataFrame of the TEP that does not contain the seed (SimulationRun), and has the sample number and the fault type at the first and last columns, respectively
    """
    tep_columns = tr_data.columns

    assert tep_columns[0] == 'sample' or tep_columns[
      0] == 'sample_id', 'The TEP dataframes passed to the class should have the sample number and the fault type at the first and last columns, respectively'
    assert tep_columns[
             -1] == 'faultNumber', 'The TEP dataframes passed to the class should have the sample number and the fault type at the first and last columns, respectively'

    self._process_vars_idx = slice(1,-1)  # this is used to exclude the sampleNumber and faultNumber columns from the TEP data

    self.train_df = tr_data
    self.val_df = val_data
    self.test_df = test_data

    # define the splitting paramters to create input output pairs
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.tot_width = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = arange(self.tot_width)[self.input_slice]
    self.label_slice = slice(self.tot_width - self.label_width, None)
    self.label_indices = arange(self.tot_width)[self.label_slice]

    # stroing the feature and traget variables names
    self.target_names = target_names
    # TODO: for the TEP data (specifically) the columns contain the faultNumber and sampleNumber variables
    # I can either remove them form the dataframe before usign it, or I can remove them in the initilization of the dp
    self.features_names_dic = {name: i for i, name in enumerate(tr_data.columns)}
    if target_names is not None:
      self.target_names_dic = {name: i for i, name in enumerate(target_names)}

    self.sampling_rate = sampling_rate
    self.sequence_stride=seqeunce_stride
    self._time_increment = sampling_rate * 3  # since the TEP contains samples that are 3 mins apart

    self.batch_size = batch_size
    self._shuffle_flag = shuffle_flag

    self._example = None

    self.flag_imglike_data=flag_imglike_data

  def split(self, data):
    input_data = data[:, self.input_slice, :]
    output_data = data[:, self.label_slice, :]

    if self.target_names is not None:
      output_data = tf.stack([output_data[:, :, self.features_names_dic[name]] for name in self.target_names],
                             axis=-1)

    input_data.set_shape([None, self.input_width, None])
    output_data.set_shape([None, self.label_width, None])

    if self.flag_imglike_data:
      input_data=tf.expand_dims(input_data,axis=-1)


    return input_data, output_data

  def visualize_batch(self, data_batch, num_rows: int = 3, num_cols: int = 4, **kwargs):
    """
    visualizes the 2-dimensional data (time,features) as images

    :param :data_batch: batch of the tep data with the first and last columns representing the sample number and the fault type, respectively

    :param :num_rows: number of the rows in the subplot (note: num_rows*num_columns<=number of samples )

    :param :num_cols: number of columns in the subplot

    :param :kwargs: dictionary containing name value pairs
    """

    num_of_samples = tf.shape(data_batch)[0]  # get the number of sample
    if kwargs.get('figsize') is not None:
      fs = kwargs.get('figsize')
    else:
      fs = (10, 10)

    cmap = kwargs.get('cmap', None)

    widths = [4 for i in range(num_cols)]
    widths.append(1)
    fig = plt.figure(figsize=fs)  # one column is added to account for the colormap object
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols + 1, width_ratios=widths)

    aux_counter = 0
    data_index = slice(1, -1)  # this excludes the sample number and the fault type from the data
    for i in range(num_rows):
      for j in range(num_cols):
        ax = plt.subplot(gs[i, j])
        h = ax.imshow(tf.transpose(data_batch[aux_counter, :, data_index]), aspect='auto', cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Sample-' + str(aux_counter))
        if i == num_rows - 1:
          # ax.set_xlabel('Time ('+str(self._time_increment)+ 'mins)')
          pass
        aux_counter += 1

    # Add a colorbar to the right of the last subplot
    cbar_ax = plt.subplot(gs[:, -1])  # [left, bottom, width, height]
    colorbar = plt.colorbar(h, cax=cbar_ax)

    plt.show()

    return None

  def visualize_reconst_error(self, databatch, cae_model):
    reconst = cae_model(databatch)

    if databatch.dtype == tf.float64:
      databatch = tf.cast(databatch, 'float32')

    reconst_error = tf.math.abs(databatch - reconst)
    self.visualize_batch(data_batch=reconst_error, cmap='gray')

    return None

  def create_data_set(self, data: pd.DataFrame):
    data = data.values
    inshape_data = data[:,self._process_vars_idx]


    ds = keras.utils.timeseries_dataset_from_array(data=inshape_data, targets=None, sequence_length=self.tot_width,
                                                   sequence_stride=self.sequence_stride, sampling_rate=self.sampling_rate,
                                                   batch_size=self.batch_size, shuffle=self._shuffle_flag).prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.map(self.split).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

  @property
  def train_ds(self):
    return self.create_data_set(self.train_df)

  @property
  def val_ds(self):
    return self.create_data_set(self.val_df)

  @property
  def test_ds(self):
    return self.create_data_set(self.test_df)

  @property
  def example(self):
    result = getattr(self, '_example', None)
    if result is None:
      result = next(iter(self.train_ds))
      self._example = result
    return self._example