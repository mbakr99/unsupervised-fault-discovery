import pdb

import pandas as pd
from math import floor
from typing import List
from numpy import arange, ones, zeros

def tep_rdata_to_pd(tep_rdata):
  """
  take TEP data in R format and extracts the stored pd.DataFrame
  """
  keys=list(tep_rdata.keys())
  return tep_rdata[keys[0]] #extract the data frame from the RData dic


def get_tep_data_fixedsimrun(tep_data:pd.DataFrame, sim_run:int):
  """
  takes the tep pd.DataFrame and extracts all data produced by the same random seed (represented by sim_run). The resulting dataframe does not have
  a simulationRun column

  """
  assert 1 <= sim_run <= 500, 'Simulation run should be in the range [1,500]'
  column_labels_to_keep=tep_data.columns.to_list()
  column_labels_to_keep.remove('faultNumber')
  column_labels_to_keep.append('faultNumber')
  idx=tep_data[tep_data['simulationRun']==sim_run].index
  df_sim_run=tep_data.loc[idx,column_labels_to_keep] #this will include all the columns from "sample" to 'xmv_11'
  return df_sim_run


def get_single_fault_data(tep_data:pd.DataFrame, fault_number:int):
  """
  takes a tep pd.DataFrame and extracts all data related to a specific fault (represented by fault_number)
  """

  column_labels_to_keep=tep_data.columns.to_list()
  if 'simulationRun' in column_labels_to_keep:
    column_labels_to_keep.remove('simulationRun')

  column_labels_to_keep.remove('faultNumber')
  column_labels_to_keep.append('faultNumber')

  assert 1 <= fault_number <= 20, 'Fault number should be between 1 and 20'
  return tep_data.loc[tep_data['faultNumber']==fault_number,column_labels_to_keep]


def unify_tepdata_columns_order(tep_data:pd.DataFrame):
  """
  sets the columns of the tep dataframe to such that the sample number and the fault number are the first and last columns, respectively
  .It also removes the simulationRun column

  :param tep_data: pandas dataframe storing the TEP data
  :return:
  """

  column_labels_to_keep = tep_data.columns.to_list()
  if 'simulationRun' in column_labels_to_keep:
    column_labels_to_keep.remove('simulationRun')

  column_labels_to_keep.remove('faultNumber')
  column_labels_to_keep.remove('sample') #for some reason, slicing 'sample' column does not work as expected. Thus; I am removing it and creating a similar column to work around the problem
  column_labels_to_keep.append('faultNumber')

  temp=tep_data.loc[:, column_labels_to_keep]
  n=len(tep_data)
  sample_id=arange(1,n+1)
  temp.insert(0,'sample_id',sample_id)

  return temp


def split_tep_data(tep_data:pd.DataFrame, split_frac:float):
  """
  splits tep pd.DataFrame into two segments defined by the split parameter split_frac. Since this is time series data
  no shuffling is applied to preserve the temporal relationship between the samples

  :param  tep_data: pd.DataFrame storing the tep data
  :param split_frac: a float in the range [0,1] controlling the size of the two splits. For example split_frac=0.2 result in two segments
  containing  20% and 80% of the data, respectively
  """

  n=len(tep_data)
  split_idx=floor(n*split_frac)


  return tep_data.iloc[:split_idx,:], tep_data.iloc[split_idx:,:]


def normalize_data(data:pd.DataFrame,kept_columns: List[str]=['sample_id','faultNumber']):

  #get the position index of the columns we want to keep
  pos_idx=[idx for idx,column_name in enumerate(data.columns) if column_name in kept_columns]
  #pop the columns that will not go normalization aside
  kept_columns_intact=[]
  for i in range(len(kept_columns)):
    kept_columns_intact.append(data.pop(kept_columns[i]))


  #normalize the data
  mu,std=data.mean(),data.std()
  norm_data=(data - mu) / std

  #insert back the intact columns
  for i in range(len(kept_columns)):
    data.insert(pos_idx[i],kept_columns[i],kept_columns_intact[i]) #this line restore the changes applied to the global variables
    norm_data.insert(pos_idx[i],kept_columns[i],kept_columns_intact[i])


  return norm_data, [mu, std], kept_columns


def apply_norm(data:List[pd.DataFrame], norm_params,kept_columns):
  """
  applies an std normalization defined by norm_params

  : param norm_params: list containing the normalization mean and standard deviation
  """
  pos_idx_dic = {column_name:idx for idx, column_name in enumerate(data[0].columns) if column_name in kept_columns}
  #loop over the data and extract the columns that will not undergo normalization aside
  intact_cols_list_up=[]
  for data_set in data:
    intact_cols_list_low=[]
    for kept_col in kept_columns:
      intact_cols_list_low.append(data_set.pop(kept_col))
    intact_cols_list_up.append(intact_cols_list_low)


  #normalize the data
  mu=norm_params[0]
  std=norm_params[-1]
  norm_list=[]
  for data_set in data:
    norm_list.append((data_set - mu) / std)


   #insert back the intact columns
  for i,norm_data_set in enumerate(norm_list):
    corresponding_kept_cols=intact_cols_list_up[i]
    for j,kept_col_name in enumerate(pos_idx_dic):
      data[i].insert(pos_idx_dic.get(kept_col_name),kept_col_name,corresponding_kept_cols[j]) #this is to revert the chnage made to the global 'data' pd.DataFrame
      norm_data_set.insert(pos_idx_dic.get(kept_col_name),kept_col_name,corresponding_kept_cols[j])



  return norm_list


def get_samples_fault_state(tep_data, fault_injection_time: int, sampling_period: int =3, normal_operation_flag=False):
  '''

  :param tep_data: TEP data that corresponds to a single fault type and a single simulation run
  :param fault_injection_time: The time at which the fault is introduced to the simulation in hours
  :param sampling_period: The difference in minutes between consecutive observations in minutes (it is typically three minutes)
  :return: returns the TEP pandas dataframe with a boolean column "sample_fault_state" that specifies whither  the corresponding observation
  was recorded after the fault or not
  '''

  num_samples=len(tep_data)
  samples_state = ones(shape=(num_samples,),dtype=bool)

  fault_injection_idx=int(fault_injection_time*(60/sampling_period))
  #print(fault_injection_idx)  This was just for testing purpose
  samples_state[:fault_injection_idx]=False

  if normal_operation_flag:  #in case of normal operating conditions set all samples fault state to False (indicating healthy observations)
    samples_state=zeros(shape=(num_samples,),dtype=bool)




  return  samples_state
