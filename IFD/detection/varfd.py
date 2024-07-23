import pdb

from scipy.spatial.distance import  mahalanobis
from numpy import outer, diag
from sklearn.metrics import confusion_matrix
from tensorflow import reshape

def analyze_sample(sample,mean,cov_mat,z_thresh):
    '''

    :param sample: sample of the process data
    :param mean: the mean of the predicted distribution
    :param cov_mat: the covariance of the predicted distribution
    :param z_thresh: threshold to determine if the marginal distribution is contribution to the anomaly or not

    :return z_scroe: vector of each feature z-score using the marginal distribution
    :return anomaly_source: boolean vector indicating the source of anomaly based on the z-score of each feature (remember each feature corresponds to a process variable)
    '''

    marginal_cov=diag(cov_mat)  #since the distribution is gaussian
    #pdb.set_trace()
    z_score=(sample-mean)/(marginal_cov)
    anomaly_source= z_score>=z_thresh

    return z_score,anomaly_source


def detect_fault(ss_rnn, data, mahala_thresh: float, z_thresh: float=3):
    '''
    This function uses a probabilistic ss_rnn model to detect faults
    
    :param ss_rnn: trained ss_rnn model
    :param data: sequential data of interest of size l x k (k is the number of features in each observation, l is the number of observations (time window))
    :param mahala_thresh: threshold on the Mahalanobis distance
    :param z_thresh: threshold on the z_score of the marginal distributions of the process variable to locate the source of anomaly
          
    :return m: a 1-d vector of l elements denoting the anomaly score for
    :return f: a 1-d vector of l logical elements denoting the existence of a fault for each of the l observations
    :return z: an l x k array that contains l (kx1) vectors that represent the z-score of each observation/sample
    :return c: an l x k bool array that contains l (kx1) vectors describing the features contributing to the anomaly
    '''

    data_shape=data.shape
    assert len(data_shape)==2, 'The data should be of shape [time,features]'

    l,k=data_shape

    m=[] #list to store mahalanobis distance
    f=[] #list to denote fault status
    z=[] #list to store the z-score vector of each observation/sample
    c=[] #list to store the characterizing vector of each observation/sample



    for i in range(l):

        #set fault status to zero
        fault_status=False
        #TODO: Something is wrong here. I am using the input sample in Mahalnobis distance calculations
        #the trained model expects a shape pf [batch,time,features]
        sample=data[i].reshape([1,1,-1]) #batch_size=1, time=1;
        pred=ss_rnn(sample)
        pred_mean,pred_cov_vec=pred[0],pred[1]
        #reshape pred_mean and sample
        pred_mean=reshape(pred_mean, shape=[-1, ])
        sample=reshape(sample,shape=[-1,])
        cov_mat=outer(pred_cov_vec,pred_cov_vec)

        #compute Mahalanobis distance
        mahala_dist=mahalanobis(pred_mean,sample,cov_mat)

        if mahala_dist>=mahala_thresh:
            fault_status=True
            z_score_vec_i,anoamly_source_i=analyze_sample(sample,pred_mean,cov_mat,z_thresh=z_thresh)
            z.append(z_score_vec_i)
            c.append(anoamly_source_i)


        m.append(mahala_dist)
        f.append(fault_status)

    return m,f,z,c



def align_target_pred_seq(target, pred, time_overhead):
    '''
    This function returns the observations that exist in both the target and prediction sequences

    :param target: target sequence
    :param pred: prediction sequence
    :param time_overhead: the number of observations the model uses to make a prediction (some models uses three
    consecutive observations to produce a prediction
    :return: two pruned iterables
    '''
    return target[time_overhead:],pred[:-time_overhead]


def get_fault_detection_metrics(target_fault_state,predicted_fault_state):
    '''

    :param target_fault_state: a boolean vector of the same length as the sequence under consideration that indicates the existence of
    faults (True) in each observation of the sequence
    :param predicted_fault_state: a boolean vector of the same length as the sequence under consideration that indicates the predicted
    fault states for each observation
    :return: returns the false positive rate fpr, false negative rate fnr, and other fault detection performance metrics
    '''
    tn, fp, fn, tp=confusion_matrix(target_fault_state,predicted_fault_state).ravel()
    #pdb.set_trace()
    fpr= fp/(fp+tn)
    fnr=fn/(fn+tp)
    sensitivity=tp/(tp+fn)
    specifity=tn/(tn+fp)
    accuracy=(tn+tp)/(tn+tp+fn+fp)

    return [fpr,fnr,sensitivity,specifity,accuracy]

