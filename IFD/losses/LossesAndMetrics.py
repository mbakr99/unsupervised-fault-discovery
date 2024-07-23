import pdb
import tensorflow as tf
import tensorflow_probability as tfp
from numpy import concatenate, arange
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter



def mean_abs_err_var(target,pred):
    #TODO let the prediction be of shape [2,batch_size,time_steps,output_dim]
    expected_pred_shape=(2,None,None,None)
    expected_target_shape=(None,None,None)

    if len(tf.shape(pred)) <len(expected_pred_shape):
        #in this case the last axis has been collapsed
        pred = tf.stack([pred], axis=-1)

    assert len(expected_target_shape)==len(tf.shape(target)),f"Expected a target of shape {None,None,None}"
    assert len(expected_pred_shape)==len(tf.shape(pred)),f"Expected a prediction of shape {(None,2,None,None)}"
    #assert tf.shape(pred)[0]==tf.constant(2),f"Expected the prediction to have two dimensions at axis {1}" TODO This throws an error when in graph mode for some reason, check why and fix it


    pred_m=pred[0] #get the distribution mean
    return tf.reduce_mean(tf.abs(target-pred_m)) #mean absolute error


def mean_square_error_var(target,pred):
    expected_pred_shape = (2,None, None, None)
    expected_target_shape = (None, None, None)

    if len(tf.shape(pred)) <len(expected_pred_shape):
        #in this case the last axis has been collapsed
        pred = tf.stack([pred], axis=-1)

    assert len(expected_target_shape) == len(tf.shape(target)), f"Expected a target of shape {None,None,None}"
    assert len(expected_pred_shape) == len(tf.shape(pred)), f"Expected a prediction of shape {(2,None, None, None)}"
    #assert tf.shape(pred)[0] == 2, f"Expected the prediction to have two dimensions at axis {1}" TODO


    pred_m = pred[0]  # get the distribution mean
    return tf.reduce_mean((target - pred_m)**2)  # mean squared error


#@tf.function()
def variational_log_likelihood(y_true,y_pred):
  #idealy y_pred would be a set of distribution object (num_time_steps) where distribution.log_prob() can be used directly to compute the loss
  #however, since I am not able to store the objects when operating in graph mode, y_pred is storing the parameters of the Normal distribution (this restricts the usage of this specific loss function to Normal distributions)
  #Thus, the shape of y_pred is [2,batch_size,time_steps,output_dim], the two is a result of storing two parameters, namely, loc and scale (mean,std)
  expected_len=len([2,None,None,None])
  if len(tf.shape(y_pred))<expected_len: #TODO this if condition might cause problem in graph mode (check before using)
    y_pred=tf.stack([y_pred],axis=-1) #TODO : the prediction [2,batch_size,time_steps,output_dim], however, when the output_dim=1, the prediction get squeezed for some reason
  else:
      y_pred=y_pred #NOTE this is just to avoid errors during autograph tracing

  y_true=tf.transpose(y_true,perm=[1,0,2]) #this will be of shape [time_step,batch_size,output_dim]

  y_pred=tf.transpose(y_pred,perm=[2,0,1,3]) #this will be of shape [time_steps,2,batch_Size,output_dim]
  num_time_steps=tf.shape(y_true)[0]
  loss=tf.constant(0.0,dtype='float32')



  for time_step in tf.range(num_time_steps):
    #pdb.set_trace()
    loc_i=y_pred[time_step,0,:,:]
    scale_i=y_pred[time_step,1,:,:]
    dist_i=tfp.distributions.Normal(loc=loc_i,scale=scale_i)
    loss_i=dist_i.log_prob(y_true[time_step])
    loss+=tf.reduce_sum(loss_i)
  return -loss


def get_mdl_perf_seq(dataset,mdl):
    #dataset has to contain the sample id information. Why? Because the only two ways to keep track of the samples after creating a dataset is to account for the effects of the dataset
    #paramters in the resulting shift of the sample id, or to keep a sample id columns that stores the info as the transformations cause by the dataset creation are taking place.
    #dataset is a TEP dataset that contains a sampleid and faultNumber columns at index 0 and 53, respectively
    # This function takes a "visualization" datapipe and the trained (probabilistic) model. Then, it returns the model prediction and the process variables
    # as a 2d array (time_window,process_vars).
    # The "visualization" datapipe uses a stride of a size equal to the "sequence_length" to result in time series data with non-overlapping indices
    #full_target_seq is a lxk array that corresponds to the process measurement over the full period. Why I am doing this? Because when I want to plot the model prediction vs the target
    #I want to show every possible detail. Note: There is no guarantee that I am able to get the full sequence from the dataset (actually I don't think it is possible)
    pred_l = []
    sample_id_l=[]
    #target_l = []
    process_vars_slice=slice(1,-1)
    sample_id=0


    dummy_var = 1

    for i, j in iter(dataset):
        sample_id_l.append(j[:,:,sample_id])
        pred = mdl(i[:,:,process_vars_slice]).numpy()
        #pdb.set_trace()
        pred_l.append(pred)
        #target_l.append(i.numpy()) #TODO: This is not the full sequence

        if dummy_var < 2:
            num_variables = j[:,:,process_vars_slice].shape[-1]
            dummy_var += 1

    num_of_output_per_pred=pred.shape[0]


    if num_of_output_per_pred==2:  #when the model outputs a distribution (mean,cov) instead of a deterministic prediction
        pred_arr = concatenate(pred_l, axis=1)
        pred_arr=pred_arr.reshape([2, -1, num_variables])
    else:
        pred_arr = concatenate(pred_l, axis=0)
        pred_arr = pred_arr.reshape([-1, num_variables])


    sample_id_arr=concatenate(sample_id_l,axis=0) #this results in [tot_batches ,in_width] shape
    sample_id_arr=concatenate(sample_id_arr,axis=0) #this results in [in_width*tot_batches ,] shape
    #target_arr = concatenate(target_l, axis=0)

    return pred_arr, sample_id_arr#target_arr.reshape([-1, num_variables])



def plot_process_vars_and_pred(prediction_seq,target_seq,pred_samples_id,process_vars_name,fault_injection_sample,ylims,mean_plot_alpha,uncertainty_bound_alpha,uncertainty_factor=3):
    #prediction_seq is expected to be of shape [2,timesteps, features]
    #target_seq is  expected to be of shape [timesteps, features]
    num_variables=target_seq.shape[-1]
    num_cols=9
    num_rows=ceil(num_variables/num_cols)

    aspect_ratio=3
    width_ratios = [1 for i in range(num_cols)]
    height_ratios=[1/aspect_ratio for i in range(num_rows)]

    fig=plt.figure(figsize=(50,25))
    g_spec=gridspec.GridSpec(num_rows,num_cols,height_ratios=height_ratios,width_ratios=width_ratios)

    axes=[]
    k=0



    num_samples = len(target_seq)
    full_timestamp = arange(1, (num_samples + 1))

    pred_samples_id=pred_samples_id.astype('int32')-1 #the process of creating the timeseries converted the sampleid from int to float
    #pdb.set_trace()
    pred_timestamp = full_timestamp[pred_samples_id] #TODO: Make sure that the index is consistent with zero-based indexing (I might have to deduct 1)/ Turns out I had to


    #set a color map
    colors = plt.cm.get_cmap('tab20c', 20)

    #define a fontdic for the y-axis labels
    ylabels_fd={'weight':'bold', 'size':15, 'family':'sans-serif'}
    xlabels_fd={'weight':'bold', 'size':15, 'family':'sans-serif','rotation':45}
    ticks_fd = {'weight': 'bold', 'size': 15, 'family': 'sans-serif','rotation':45}
    title_fd={'weight':'bold','size':20,'family':'sans-serif'}
    leg_fd={'weight':'bold','size':20,'family':'sans-serif'}

    var_model_flag= prediction_seq.shape[0]==2 #since a variational model will output (in my case) a mean and cov vectors (hence 2)

    counter=0
    if var_model_flag:

        for i in range(num_rows):
            for j in range(num_cols):

                if (k+1)<= num_variables:
                    pred_mean_var_k=prediction_seq[0,:,k]
                    pred_std_var_k=prediction_seq[-1,:,k]
                    upper_bound=pred_mean_var_k+uncertainty_factor*pred_std_var_k
                    lower_bound=pred_mean_var_k-uncertainty_factor*pred_std_var_k
                    target_var_k=target_seq[:,k]




                    ax_k=plt.subplot(g_spec[i,j])
                    ax_k.plot(full_timestamp,target_var_k,label='Process measurement',linewidth=3,color=colors(6))
                    ax_k.plot(pred_timestamp,pred_mean_var_k,alpha=mean_plot_alpha,label='Prediction mean',linewidth=3,color=colors(0))
                    ax_k.fill_between(pred_timestamp, lower_bound, upper_bound, color=colors(3),alpha=uncertainty_bound_alpha,label='Confidence-interval')

                    if ylims: #if the limits is a non-empty list
                        ax_k.set_ylim(bottom=ylims[0],top=ylims[-1])

                    ax_k.set_yticklabels(ax_k.get_yticks(),fontdict=ylabels_fd)
                    if j != 0: #remove the y-axis ticks except for thefirst plot in the row
                        ax_k.set_yticks([])





                    if fault_injection_sample is not None:
                        ax_k.axvline(x=fault_injection_sample, color='red')

                    # if i == 0 and j == num_cols - 1:  TODO: I replaced this with a legend placed below the figures
                    #     ax_k.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

                    if (k+1) > (num_variables-num_cols):
                        ax_k.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        ax_k.set_xticklabels(ax_k.get_xticks(),fontdict=xlabels_fd)
                    else:
                        ax_k.set_xticks([])


                    ax_k.set_title(process_vars_name[counter],fontdict=title_fd)
                    axes.append(ax_k)
                    counter+=1
                    k += 1 #TODO: I might be using two different variables for the same purpose

        #pdb.set_trace()
        empty_cols=tf.math.abs(num_variables-num_rows*num_cols)

        h,l=ax_k.get_legend_handles_labels()
        legend_axs=fig.add_subplot(g_spec[num_rows-1,(num_cols)-empty_cols:])
        legend_axs.set_axis_off()
        legend_axs.legend(handles=h,labels=l,loc='upper center', bbox_to_anchor=(0.5, 0.5),prop=leg_fd, shadow=True, ncol=2,markerscale=1.5) #,


    else:

        for i in range(num_rows):
            for j in range(num_cols):

                if (k + 1) <= num_variables:
                    pred_mean_var_k = prediction_seq[:, k]
                    target_var_k = target_seq[:, k]

                    k += 1

                    ax_k = plt.subplot(g_spec[i, j])
                    ax_k.plot(full_timestamp, target_var_k, label='Process measurement')
                    ax_k.plot(pred_timestamp, pred_mean_var_k, alpha=mean_plot_alpha, label='Prediction mean')

                    if j != 1: #remove the y-axis ticks except for thefirst plot in the row
                        ax_k.set_yticks([])
                    if ylims:  # if the limits is a non-empty list
                        ax_k.set_ylim(bottom=ylims[0], top=ylims[-1])

                    if fault_injection_sample is not None:
                        ax_k.axvline(x=fault_injection_sample,color='red')

                    if i==0 and j==num_cols-1:
                        ax_k.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

                    ax_k.set_title(process_vars_name[counter])
                    axes.append(ax_k)
                    counter += 1


    fig.subplots_adjust(wspace=0.25, hspace=0.35)



    return axes



def visualize_model_results(dataset,full_target_seq,process_vars_name,mdl,fault_injection_sample=160,ylims=[],mean_plot_alpha=0.7,uncertainty_bound_alpha=1,conf_width=3):

    predic_seq,pred_samples_id=get_mdl_perf_seq(dataset,mdl)
    axes=plot_process_vars_and_pred(predic_seq,full_target_seq,pred_samples_id=pred_samples_id,uncertainty_factor=conf_width,fault_injection_sample=fault_injection_sample,process_vars_name=process_vars_name,mean_plot_alpha=mean_plot_alpha,uncertainty_bound_alpha=uncertainty_bound_alpha,ylims=ylims)
    return axes




def plot_tep_timeseris(data,ylims,fault_injection_sample,process_vars_name,data_labels,data_colors):
    #data is a list storing multiple timeseries data
    num_variables = data[0].shape[-1]
    num_timestamps = data[0].shape[0]
    num_cols = 9
    num_rows = ceil(num_variables / num_cols)

    aspect_ratio = 3
    width_ratios = [1 for i in range(num_cols)]
    height_ratios = [1 / aspect_ratio for i in range(num_rows)]

    fig = plt.figure(figsize=(50, 25))
    g_spec = gridspec.GridSpec(num_rows, num_cols, height_ratios=height_ratios, width_ratios=width_ratios)

    # define a fontdic for the y-axis labels
    ylabels_fd = {'weight': 'bold', 'size': 17, 'family': 'sans-serif'}
    xlabels_fd = {'weight': 'bold', 'size': 17, 'family': 'sans-serif', 'rotation': 45}
    ticks_fd = {'weight': 'bold', 'size': 15, 'family': 'sans-serif', 'rotation': 45}
    title_fd = {'weight': 'bold', 'size': 20, 'family': 'sans-serif'}
    leg_fd = {'weight': 'bold', 'size': 20, 'family': 'sans-serif'}


    axes = []
    k = 0

    counter = 0

    colors = plt.cm.get_cmap('tab20c', 20)

    full_timestamp=arange(1,num_timestamps+1)

    for i in range(num_rows):
        for j in range(num_cols):

            if (k + 1) <= num_variables:


                ax_k = plt.subplot(g_spec[i, j])
                for data_count,ts in enumerate(data):
                    ax_k.plot(full_timestamp, ts[:,k], label=data_labels[data_count], linewidth=3, color=colors(data_colors[data_count]))



                if ylims:  # if the limits is a non-empty list
                    ax_k.set_ylim(bottom=ylims[0], top=ylims[-1])

                ax_k.set_yticklabels(ax_k.get_yticks(), fontdict=ylabels_fd)
                if j != 0:  # remove the y-axis ticks except for thefirst plot in the row
                    ax_k.set_yticks([])

                if fault_injection_sample is not None:
                    ax_k.axvline(x=fault_injection_sample, color='red')

                # if i == 0 and j == num_cols - 1:  TODO: I replaced this with a legend placed below the figures
                #     ax_k.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

                if (k + 1) > (num_variables - num_cols):
                    ax_k.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax_k.set_xticklabels(ax_k.get_xticks(), fontdict=xlabels_fd)
                else:
                    ax_k.set_xticks([])

                ax_k.set_title(process_vars_name[counter], fontdict=title_fd)
                axes.append(ax_k)
                counter += 1
                k += 1  # TODO: I might be using two different variables for the same purpose

    # pdb.set_trace()
    empty_cols = tf.math.abs(num_variables - num_rows * num_cols)

    h, l = ax_k.get_legend_handles_labels()
    legend_axs = fig.add_subplot(g_spec[num_rows - 1, (num_cols) - empty_cols:])
    legend_axs.set_axis_off()
    legend_axs.legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(0.5, 0.5), prop=leg_fd, shadow=True,
                      ncol=2, markerscale=1.5)  # ,

