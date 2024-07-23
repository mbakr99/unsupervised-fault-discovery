import seaborn as sns
import pandas as pd
from tensorflow import reshape as reshape,shape,stack, TensorSpec
import pdb
import matplotlib.pyplot as plt


def visualize_var_mdl(mdl,input_data,true_target:TensorSpec(shape=[1,None,None],dtype='float32')=None,mdl_type='SsRnnVar',output_vars_dic=None,var_name=None):
    #this function assumes that the input_data has one batch only
    pred=mdl(input_data)
    if mdl_type=='SsRnnVar': #in this case pred has the shape [2,batch_size=1,time_steps,output_dim]
        pass #TODO set these branches to account for using different models that have a different shaped output (the modification should result in a shape [1,time_steps,output_dim]
    else:
        pass
    #dtermine the output feature to plot
    plot_var_idx=-1
    if var_name is not None and output_vars_dic is not None:
        plot_var_idx=output_vars_dic.get(var_name)

    #these now have shape [1,time_steps,output_dim]
    pred_std=pred[1]

    num_time_steps=shape(pred_std)[1]
    pred_std=reshape(pred_std,shape=[num_time_steps,-1])
    pred_m = reshape(pred[0],shape=[num_time_steps,-1]) #this contains all the output features corresponding to plot_var
    pred_m_plot_var=pred_m[:,plot_var_idx] #this contains the output feature corresponding to plot_
    pred_ub=pred_m[:,plot_var_idx]+3*pred_std[:,plot_var_idx]
    pred_lb=pred_m[:,plot_var_idx]-3*pred_std[:,plot_var_idx]

    if var_name is None:
        name_prefix=''
    else:
        name_prefix=var_name
    pd_columns_headings=[name_prefix+'-lower-bound',name_prefix+'-mean',name_prefix+'-upper-bound']
    data=stack([pred_lb,pred_m_plot_var,pred_ub],axis=1)

    pred_df=pd.DataFrame(data.numpy(),columns=pd_columns_headings)

    if true_target is not None:
        true_target_plot_var=true_target[:,:,plot_var_idx] #the shape of true traget
        true_target_plot_var=reshape(true_target_plot_var,shape=[num_time_steps,])
        sns.lineplot(true_target_plot_var)

    sns.lineplot(pred_df)
    plt.show()
    return pred_df
