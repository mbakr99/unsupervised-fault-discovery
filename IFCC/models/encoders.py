import pdb

import tensorflow as tf
from tensorflow import keras
from typing import List
from numpy import concatenate, arange
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter



class CustomEncoder(keras.Model):

  def __init__(self,layers : List= None, activation : str= None, output_dim : int=10):
    super(CustomEncoder,self).__init__()
    self.mdl=None
    if layers is not None: #in this case output_dim is not really used
      self.t_layers=layers
    else:
      with tf.name_scope('custom_encoder'):
        self.t_layers=[keras.layers.Conv2D(32,(3,3),activation=activation,name='conv1'),
                       keras.layers.AveragePooling2D((2,2),name='pool1'),
                       keras.layers.Conv2D(16,(2,2),activation=activation,name='conv2'),
                       keras.layers.AveragePooling2D((2,2),name='pool2'),
                       keras.layers.Flatten(name='flatten'),
                       keras.layers.Dense(output_dim,name='output_layer')]

  def call(self,input):
    #pdb.set_trace()
    x=input
    for layer in self.t_layers:
      x=layer(x)
    return x

  def get_layers_info(self,encoder_input_shape):
    #returns a list containing the following layers info #order | name | (num_filters,kernel_size) | output shape
    x=tf.random.normal(shape=[1]+encoder_input_shape,dtype='float32')
    layers_info=[]
    for i,layer in enumerate(self.t_layers):
      x=layer(x)
      #layers_info['layer-'+str(i+1)]=[layer.name,x.shape[1:]]
      #pdb.set_trace()
      try:
        num_filters,kernel_size,padding=layer.filters,layer.kernel_size,layer.padding
      except:
        num_filters,kernel_size,padding=None,None,None

      layers_info.append([i+1,layer.name,(num_filters,kernel_size,padding),x.shape[1:].as_list()])
    return layers_info


class CustomDecoder(keras.Model):

  def __init__(self,decoder_layers,target_shape):
    super().__init__()
    self.t_layers=decoder_layers
    self.target_shape=target_shape
    self._correction_flag=False

  def get_layers_info(self,decoder_input_shape):
    #returns a list containing the following layers info #order | name | (num_filters,kernel_size) | output shape
    if self._correction_flag:
      print('Note: These are the layers information after correcting the structure to match the target shape ')
    else:
      print('Note: These are the layers information before correcting the structure to match the target shape')

    x=tf.random.normal(shape=[1]+decoder_input_shape,dtype='float64') #the [1]+ is to denote a single batch
    layers_info=[]
    for i,layer in enumerate(self.t_layers):
      #pdb.set_trace()
      x=layer(x)
      #layers_info['layer-'+str(i+1)]=[layer.name,x.shape[1:]]
      #pdb.set_trace()
      try:
        num_filters,kernel_size,padding=layer.filters,layer.kernel_size,layer.padding
      except:
        num_filters,kernel_size,padding=None,None,None

      layers_info.append([i+1,layer.name,(num_filters,kernel_size,padding),x.shape[1:].as_list()])  #<TODO> add names to the layers of the decoder
    return layers_info

  def set_and_correct_decoder_structure(self,decoder_input_shape):

    self._correction_flag=True
    layers_info=self.get_layers_info(decoder_input_shape=decoder_input_shape) # <TODO> do I need to do this? Shouldn't the t_layer attribute have this information already?
    current_output_shape=layers_info[-1][-1]  #the shape has a structure of [height,width,num_channels]

    correcting_kernel_size=self._get_correcting_kernel_size(current_output_shape)
    self.t_layers+=[keras.layers.Conv2DTranspose(self.target_shape[-1],kernel_size=correcting_kernel_size)]
    print('A transposed convolutional layer has been added to modify the decoder output from shape {} to shape {}'.format(current_output_shape,self.target_shape))

    return None

  def _get_correcting_kernel_size(self,current_output_shape):
    h_goal,w_goal=self.target_shape[:-1] #ignore tge number of channels dimension
    h,w=current_output_shape[:-1]
    h_k,w_k=h_goal-h+1,w_goal-w+1

    return h_k,w_k



  def call(self,input):
    x=input
    for layer in self.t_layers:
      x=layer(x)
    return x


class Cae(keras.Model):

  def __init__(self,encoder_mdl,target_shape):
    super(Cae,self).__init__()
    self.encoder_mdl=encoder_mdl
    self.decoder_mdl=None
    self.target_shape=target_shape

  def build(self,input_shape):
    #pdb.set_trace()
    self.target_shape=input_shape[1:]
    encoder_output_shape,layers=self._get_decoder_structure(encoder_input_shape=input_shape[1:]) #the encoder_input_shape is passed to determine the model paramters
    self.decoder_mdl=CustomDecoder(decoder_layers=layers,target_shape=self.target_shape)
    self.decoder_mdl.set_and_correct_decoder_structure(decoder_input_shape=encoder_output_shape) #this is the result of not defining the decoder input layer (for flexibility)

  def _get_decoder_structure(self, encoder_input_shape:List[int]):
    #order | name | (num_filters,kernel_size,padding) | output shape
    output_shape_info,conv_info,name_info,order_info=-1,-2,-3,-4 #<TODO> I might substitue this with enum datatype
    last_layer_info=-1
    last_dim=-1
    #this method assumes that the encoder has the last two layers as flatten and dense layers, and that the structure is composed by stacking [conv,pooling] layers over and over
    layers=[]
    encoder_info=self.encoder_mdl.get_layers_info(encoder_input_shape=encoder_input_shape)
    #configuration of the dense and reshaping layer (decoder first two layers)
    encoder_output_shape=encoder_info.pop(last_layer_info)[output_shape_info]
    layers.append( keras.layers.Dense(encoder_info.pop(last_layer_info)[output_shape_info][-1],activation='relu',name='dense') )
    layers.append( keras.layers.Reshape(target_shape=encoder_info[last_layer_info][output_shape_info],name='reshape') )
    #configuration of the transposed conv and upsampling (decoder middle layers)
    encoder_info.reverse()
    num_hidden_layers=len(encoder_info)
    #create two variables that help identify the layer type
    anti_pool_id=num_hidden_layers/2
    anti_conv_id=num_hidden_layers/2
    for i,info in enumerate(encoder_info):

      if i % 2==0: #corresponds to pooling <TODO>: For now the pooling is assumed to be of size (2,2). Change this so that the pool window is variable
        layer_name='anti-pooling-'+str(int(anti_pool_id))
        new_layer=keras.layers.UpSampling2D(size=(2,2),name=layer_name)
        anti_pool_id-=1

      else:
        _,kernel_size,padding_type=info[conv_info]  #I added the padding info to enable using deeper networks
        if i+1<num_hidden_layers:
          num_filters=encoder_info[i+1][output_shape_info][last_dim]
        else:
          num_filters=self.target_shape[-1]

        layer_name='anti-conv-'+str(int(anti_conv_id))
        new_layer=keras.layers.Conv2DTranspose(filters=num_filters,kernel_size=kernel_size,padding=padding_type,name=layer_name,activation='relu')#modify to set activation based on encoder
        anti_conv_id-=1

      layers.append(new_layer)

    return encoder_output_shape,layers

  def call(self,input):
    return self.decoder_mdl(self.encoder_mdl(input))





class CustomDecoderLstm(keras.layers.Layer): #TODO: This is similar to the LSTM_Decoder class (the has more DOF). I have to choose between one of the two implementations

  def __init__(self,decoder_layers,custom_lstm):
    super().__init__()
    self.t_layers=decoder_layers
    self.lstm=custom_lstm

  def call(self,x):
    # convert the latent representation to an image-like structure (recovering the temporal correlations)
    x=self.lstm(x) #the shape of the custom_lstm output is [time,batch,features]. This need to be converted to [batch,time,features,1]
    #x=tf.transpose(x,perm=[1,0,2])
    x=tf.expand_dims(x,axis=-1)
    for layer in self.t_layers:
      x=layer(x)


    return tf.squeeze(x)


class AutoEncoder(keras.Model):

  def __init__(self,encoder,decoder,**kwargs):
    super().__init__(**kwargs)
    self.encoder=encoder
    self.decoder=decoder

  def call(self,x):

    return self.decoder(self.encoder(x))


class RnnLstmAutoEncoder(AutoEncoder):

  def __init__(self,rnn_encoder,lstm_decoder,**kwargs):
    super().__init__(encoder=rnn_encoder,decoder=lstm_decoder,**kwargs)

  def call(self,x):
    _,latent_repr=self.encoder(x)
    return self.decoder(latent_repr)



class LSTM_Decoder(keras.Model):

  def __init__(self,target_shape,units,activation,dropout=0.1,**kwargs):

    super().__init__(**kwargs)

    self.ar_width=target_shape[0]
    self.num_features=target_shape[-1]


    self.hidden_state_dim=units
    self.lstm_cell=keras.layers.LSTMCell(units=units,activation=activation,dropout=dropout,name='lstm')
    self.output_layer=keras.layers.Dense(units=self.num_features,name='output_dense',activation='tanh',bias_initializer='random_normal')



  def build(self,input_shape):
    latent_dim=input_shape[-1]
    self.in_dense_layer=keras.layers.Dense(input_dim=latent_dim,units=self.num_features,activation='linear',name='reshape_dense',bias_initializer='random_normal') #TODO: what activation function fits this the best
    return None



  def call(self,x):

    batch_size=tf.shape(x)[0]
    reconst=tf.TensorArray(size=self.ar_width,dtype=x.dtype)
    s1, s2 = self.lstm_cell.get_initial_state(batch_size=batch_size, dtype=x.dtype)

    x=self.in_dense_layer(x)
    print(x.shape)
    for i in tf.range(self.ar_width):
      #pdb.set_trace()
      temp,states=self.lstm_cell(x,[s1,s2])
      x=self.output_layer(temp)
      s1,s2=states
      reconst=reconst.write(i,x)


    return tf.transpose(reconst.stack(),perm=[1,0,2])




def train_cae(model,datapipe,loss_fn,optimizer,metric,**kwarg):



  epochs_temp=kwarg.get('num_epochs',None)
  EPOCHS= epochs_temp if epochs_temp is not None else 10

  tr_metric=metric
  val_metric=metric

  train_ds=datapipe.train_ds
  val_ds=datapipe.val_ds

  num_tr_batches=datapipe.train_ds.cardinality().numpy()
  num_val_batches=datapipe.val_ds.cardinality().numpy()

  tr_loss=[]
  val_loss=[]

  for epoch in range(EPOCHS):

    for tr_step,(tr_batch,_) in enumerate(train_ds):
      #training step
      #model forward pass
      with tf.GradientTape() as tr_tape:
        reconstruction=model(tr_batch)
        loss=loss_fn(tr_batch,reconstruction)

      #calculate gradients
      grads=tr_tape.gradient(loss,model.trainable_weights)
      optimizer.apply_gradients(zip(grads,model.trainable_weights))
      #update training metric

      tr_metric.update_state(tr_batch,reconstruction)


    #validation step (evaluate model on the validations set)
    for val_step,(val_batch,_) in enumerate(val_ds):

        reconstruction=model(val_batch)
        val_metric.update_state(val_batch,reconstruction)


    tr_loss.append(tr_metric.result())
    val_loss.append(val_metric.result())

    template = 'Epoch {}, Loss: {}, Val Loss: {}'

    #print(template.format(epoch,tr_metric.result(),val_metric.result()))


    #reset both the training and validation metrics
    tr_metric.reset_states()
    val_metric.reset_states()

  return {'train_loss':tr_loss,'val_loss':val_loss}



def get_target_and_reconstruction_seq(dataset,mdl):

  # TODO: Fix the code to be able to deal with the fact that mdl cab have one or two outputs based on its type
  reconst_l=[]
  target_l=[]

  for i,_ in iter(dataset):
    reconst=mdl(i)

    if len(i.shape)>3:
      i=tf.squeeze(i,axis=-1)

    if len(reconst.shape)>3:
      reconst=tf.squeeze(reconst,axis=-1)

    reconst_l.append(reconst)
    target_l.append(i)


  num_of_variables=i.shape[-1]

  target_arr=concatenate(target_l, axis=0)
  reconst_arr=concatenate(reconst_l, axis=0)


  return reconst_arr.reshape([-1,num_of_variables]),target_arr.reshape([-1,num_of_variables])


def visualize_reconstruction(dataset,mdl,signal_names,ylims):
  # set a color map



  colors = plt.cm.get_cmap('tab20c', 20)

  # define a fontdic for the y-axis labels
  ylabels_fd = {'weight': 'bold', 'size': 15, 'family': 'sans-serif'}
  xlabels_fd = {'weight': 'bold', 'size': 15, 'family': 'sans-serif', 'rotation': 45}
  title_fd = {'weight': 'bold', 'size': 20, 'family': 'sans-serif'}
  leg_fd = {'weight': 'bold', 'size': 20, 'family': 'sans-serif'}





  r,t=get_target_and_reconstruction_seq(dataset,mdl)

  num_variables = t.shape[-1]
  num_cols = 9
  num_rows = ceil(num_variables / num_cols)

  aspect_ratio = 3
  width_ratios = [1 for i in range(num_cols)]
  height_ratios = [1 / aspect_ratio for i in range(num_rows)]

  fig = plt.figure(figsize=(50, 25))
  g_spec = gridspec.GridSpec(num_rows, num_cols, height_ratios=height_ratios, width_ratios=width_ratios)

  axes = []
  k = 0

  num_time_steps = t.shape[0]
  time = arange(0, num_time_steps) + 1


  counter=0
  for i in range(num_rows):
    for j in range(num_cols):

      if (k + 1) <= num_variables:
        reconst_mean_var_k = r[:, k]
        target_var_k = t[:, k]



        ax_k = plt.subplot(g_spec[i, j])
        ax_k.plot(time, target_var_k, label='True',color=colors(6),linewidth=3.5)
        ax_k.plot(time, reconst_mean_var_k, label='Reconstructed',color=colors(0),linewidth=3.5)

        if ylims:  # if the limits is a non-empty list
          ax_k.set_ylim(bottom=ylims[0], top=ylims[-1])


        ax_k.set_yticklabels(ax_k.get_yticks(), fontdict=ylabels_fd)
        if j != 0:  # remove the y-axis ticks except for thefirst plot in the row
          ax_k.set_yticks([])

        ax_k.set_title(signal_names[counter],fontdict=title_fd)


        if (k + 1) > (num_variables - num_cols):
          ax_k.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          ax_k.set_xticklabels(ax_k.get_xticks(), fontdict=xlabels_fd)
        else:
          ax_k.set_xticks([])

        # if i == 0 and j == num_cols - 1:
        #   ax_k.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        axes.append(ax_k)

        counter += 1
        k += 1

  empty_cols = tf.math.abs(num_variables - num_rows * num_cols)

  h, l = ax_k.get_legend_handles_labels()
  legend_axs = fig.add_subplot(g_spec[num_rows - 1, (num_cols) - empty_cols:])
  legend_axs.set_axis_off()
  legend_axs.legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(0.5, 0.5), prop=leg_fd, shadow=True,
                    ncol=2, markerscale=1.5)
  fig.subplots_adjust(wspace=0.25, hspace=0.35)
  return axes


def visualize_reconstruction_multiple_aes(dataset,mdls,mdl_names,colors,signal_names):

  r_list=[]
  _,t=get_target_and_reconstruction_seq(dataset,mdls[-1]) #the target data is the same regardless of the model
  for mdl in mdls:
    r_i,_=get_target_and_reconstruction_seq(dataset,mdl)
    r_list.append(r_i)



  num_variables = t.shape[-1]
  num_cols = 9
  num_rows = ceil(num_variables / num_cols)

  aspect_ratio = 3
  width_ratios = [1 for i in range(num_cols)]
  height_ratios = [1 / aspect_ratio for i in range(num_rows)]

  fig = plt.figure(figsize=(50, 25))
  g_spec = gridspec.GridSpec(num_rows, num_cols, height_ratios=height_ratios, width_ratios=width_ratios)

  axes = []
  k = 0

  num_time_steps =t.shape[0]
  time = arange(0, num_time_steps) + 1


  counter=0
  for i in range(num_rows):
    for j in range(num_cols):


      if (k + 1) <= num_variables:
        ax_k = plt.subplot(g_spec[i, j])
        for idx,r in enumerate(r_list):
          reconst_mean_var_k = r[:, k]

          if idx<1: #plot the target data just once
            target_var_k = t[:, k]
            ax_k.plot(time, target_var_k,color='tab:blue', label='True')

          ax_k.plot(time, reconst_mean_var_k, color=colors[idx],label='Reconstructed: '+mdl_names[idx])
          ax_k.set_title(signal_names[counter])

        k += 1
        counter += 1

        if i == 0 and j == num_cols - 1:
          ax_k.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        axes.append(ax_k)
  fig.subplots_adjust(wspace=0.25, hspace=0.35)
  return axes

