
# Unsupervised Fault Discovery 

Industrial systems require efficient and rapid corrective actions when faults occur. A fault is an abnormality that causes deviation from an expected behavior. When not resolved in a timely manner, faults can damage the equipment and lead to a loss of money and time. For example, a broken temperature sensor can indicate that a component's temperature is normal when it is off the chart! Sometimes, the effect of fault might be catastrophic, such as in the case of the [Fukushima](https://en.wikipedia.org/wiki/Fukushima_nuclear_accident) and [Deepwater Horizon](https://en.wikipedia.org/wiki/Deepwater_Horizon) incidents.

<p align="center">
  <img src="https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/Fukushima%20Incident.jpg" alt="Image Fuku" width="45%"/>
  <img src="https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/Deepwater%20Horizon%20Explosion.jpg" alt="Image Deep" width="45%"/>
</p>

Fault diagnosis is concerned with addressing the faults that a system might encounter by consistently monitoring the process and generating alarms and relative information to the operators. 
With the evolution of sensor technology, data storage, and computational power. Data-driven fault diagnosis methods have started to replace classical methods, which tend to be impractical when the industrial process of interest is complex. Data-driven methods rely **solely** on historical process data to learn how to make appropriate decisions. This can include but is not limited to, detecting when a fault is present, determining the source of the fault, or classifying the fault categories. The best of these is mapping the fault into its corresponding type, as this enables the operator to know everything about the fault before taking action. In fact, once the type is known, the operator can follow a predetermined procedure set by the field expert to address the fault optimally. ***However***, a set of labeled data is needed to train a classification algorithm, which is impractical and not realistic to obtain in real-life scenarios. ***Why?***, because it indicates that the process was allowed to operate under fault conditions for a considerable time to collect the data! 

This work presents a method that uses only Normal Operating Conditions (NOC) data to discover new faults. This is achieved through the following steps:
-  Detect faults
-  Extract descriptive features
-  Differentiate between faults
-  Find out which process variables relate the most to each discovered fault type (fault localization)

The proposed work is tested on the Tennessee Eastman Process (TEP). The full analysis is provided in this [Jupyter notebook](https://github.com/mbakr99/unsupervised-fault-discovery/blob/e5830f0c26041d850854d53189ec85d23f97779b/FCT_6.ipynb). This file provides a high-level description of the method. It also provides a discussion and interpretation of the results. 

## Detecting faults:
The proposed work relies on capturing the behavior of the process during NOC. This is done using two probabilistic models that capture two aspects of the NOC behavior. First, a temporal probabilistic model is used to capture the temporal relation between consecutive process measurements. This model reports a fault when the NOC temporal behavior is violated. The second is a static distribution to capture the correlation and the range information of the process variables during NOC. This model is concerned with how a single observation fits the trend learned from the NOC data. It reports a fault when the NOC static characteristics are violated. 
#### Temporal model:
A Probabilistic Recurrent Neural Network (PRNN) of a custom structure was created using [Tensorflow](https://www.tensorflow.org) and [Tensorflow probaility](https://www.tensorflow.org/probability)  and used to capture the NOC temporal behavior. The modelt utilizes the latest measurement to predict the distribution of the future one. The actual measurement is then compared to the distribution, and a fault is reported based on "how well it conforms to the distribution". The NOC data was partitioned into training and testing sets. The training set was normalized using standard normalization, and the normalization parameters were saved to normalize the model inputs at testing and deployment. More details can be found in this [FCT_6.ipynb](https://github.com/mbakr99/unsupervised-fault-discovery/blob/e5830f0c26041d850854d53189ec85d23f97779b/FCT_6.ipynb). The prediction of the model on the testing set vs the actual measurements is shown below 


![Figure RNN results](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/pred_val.png)          

Note that the temporal distribution captures the NOC behavior accurately as the actual measurments fall well within the confidence interval of the prediction at each time step.
A description of the process variables that appear in the above figure can be found in this [source paper](https://doi.org/10.1016/0098-1354(93)80018-I). 

#### Static model:
A multivariate gaussian distribution was used to capture the static charechterstics of the NOC data. Unlike the previous model, this model does not change with time. The following displays the model for the first three process variables. Notice how the actual NOC data distributin matches the model. 

![Figure joint](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/joint_dist.png)

#### Combining both models:
An anomaly score is a value that indicates the likelihood of the presence of a fault. The Mahalanobis distance is a measure of how far an observation is from a distribution. This work proposes an anomaly score that combines both the temporal and static characteristics of normal operating conditions (NOC). It achieves this by fusing the Mahalanobis distance of a process observation with respect to both the temporal and static models using the following equation

![Eq](https://latex.codecogs.com/svg.image?\begin{equation}\label{eq:anom_score}AS=\alpha\eta_Td_M^T&plus;(1-\alpha)\eta_Jd_M^J\end{equation}) 

where ![alpha](https://latex.codecogs.com/svg.image?\alpha) is an importance weight in the range [0,1], ![eta_T](https://latex.codecogs.com/svg.image?\eta_{T}) and ![eta_J](https://latex.codecogs.com/svg.image?\eta_{J}) are scaling factors, ![d_T](https://latex.codecogs.com/svg.image?d_M^T) and ![d_J](https://latex.codecogs.com/svg.image?d_M^J) denote the Mahlanobis distances to the temporal and static models, respectively. 
The importance of this novel anomlay score is two folds. First, by leveragin both the temporal and static characteristics of the process, the fault detection is enhanced. This becomes clear for faults the don't violate the temporal charecterstics but are out of the NOC variables range, such as fault 5 in the TEP. Fault 5 is a controllable fault, meaning that the control system is able to reject the disturbance causing the fault, which prevents the process from violating the NOC behavior. However, one of the manipulated variables -XMV(11)- remains at a higher level than that of NOC. This is shown below

![Figure f5](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/f5_fd.png)

On the other hand, relying solely on the static aspect can lead to an increased number of false alarms. This occurs when the process is operating normally, but the fault diagnosis system incorrectly reports faults.

If only the temporal aspect of the NOC behavior was cosidered, the fault will not be detected as shown below 

![Figure as-f5](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/as_f5.png)

#### Fault detection results:     

## The fault detection performance of the proposed model

The last column contains the results of [Sun et al. (2020)](https://doi.org/10.1016/j.compchemeng.2020.106991) for comparison. All values, except the ones in parentheses, represent the model detection accuracy. The FPR is shown for faults f3, f9, and f15.

| Fault Type    | FPR    | FNR    | Accuracy | Sun et al.   |
|---------------|--------|--------|----------|--------------|
| **f1**        | 0.625  | 0.125  | 99.79    | *99.75*      |
| **f2**        | 1.875  | 1.375  | 98.54    | *99.00*      |
| f3            | 1.875  | 97.88  | 18.13    | (*5.00*)     |
| **f4**        | 1.250  | 0.125  | 99.69    | *100.00*     |
| f5            | 1.250  | 0.000  | 99.79    | *100.00*     |
| **f6**        | 0.000  | 0.000  | 100.00   | *100.00*     |
| **f7**        | 0.000  | 0.000  | 100.00   | *100.00*     |
| f8            | 0.625  | 2.125  | 98.13    | *98.12*      |
| f9            | 3.750  | 96.750 | 18.750   | (*5.00*)     |
| f10           | 0.625  | 16.750 | 85.94    | *87.38*      |
| **f11**       | 1.875  | 24.750 | 79.06    | *74.75*      |
| f12           | 1.250  | 0.125  | 99.69    | *99.75*      |
| **f13**       | 0.625  | 4.750  | 95.94    | *95.75*      |
| **f14**       | 0.000  | 0.000  | 100.00   | *100.00*     |
| f15           | 0.625  | 95.875 | 20.00    | (*7.12*)     |
| f16           | 2.500  | 16.000 | 86.250   | *90.38*      |
| **f17**       | 1.250  | 5.000  | 95.63    | *96.13*      |
| f18           | 1.875  | 10.000 | 91.35    | *90.63*      |
| **f19**       | 0.625  | 20.500 | 82.81    | *88.25*      |
| **f20**       | 0.625  | 13.375 | 88.75    | *78.63*      |
| f21           | 2.500  | 52.125 | 56.15    | *48.00*      |


The average fault detection accuracy, excluding faults f3​, f9​, and f15​, is 91.96%, which is slightly higher than the 91.07% reported by [Sun et al. (2020)](https://doi.org/10.1016/j.compchemeng.2020.106991). Additionally, for  f3​, f9​, and f15, the proposed model achieves a significantly lower false positive rate (FPR), with reductions of 62.5% for f3​ and 91.22% for f15​. This reduction is attributed to the better generalization of the proposed model and the novel anomaly score.

## Extracting features + Differentiating between faults:
Extracting good features is crucial for distinguishing between different faults. Since no labels exist, the concept of "different" is vague and may need to be determined by field experts. However, a good starting point is to consider faults that have a distinct impact on process behavior as "different." To achieve this, the effect of the fault on process behavior is first isolated by subtracting the nominal NOC behavior from the process observations. This is accomplished using the predictions of the trained PRNN model and the static model. The difference between the process measurements and the measurements predicted by the PRNN or static model is referred to as residual data. By applying manual and deep feature extraction methods to the residual data, different faults can be separated, as demonstrated in the left side of the figures below. The left column shows the [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) extracted features color-coded using the true fault label. By clustering the extracted features using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), one may be able to discover the underlying fault type. The clustering results are shown in the right column below.

   

   
![lat_clust](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/t-sne_lat.png)

![stat_resid_clust](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/t-sne_stat_resid.png)

![lat_clust](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/t-sne_temp_resid.png)

In the figures above, each row represents a feature extraction method:
- **The first row**: These features are extracted using deep learning. The PRNN model is augmented by a Convolutional Neural Network (CNN) decoder and the model is trained using semi-supevised learning to minimize the autoencoder reconstruction error. *Note: The PRNN model is trained on minimizing both the prediction and reconstruction errors using a multiobjective loss*.
- **The second row**: These represent the mean amplitude and the standard deviation features of the process variables. The features are extracted from the residual data using the static model as a representative of the NOC behavior (static residual).
- **The third row**: These represent the mean amplitude and the standard deviation features of the process variables. The features are extracted from the residual data using the temporal model (PRNN) as a representative of the NOC behavior (temporal residual).

### Improving the results using a tribal-based consensus algorithm 
How can the results of clustering three different sets of features be used to improve the final decision? An algorithm that mimics the social behavior of tribes in forming or ending social connections is developed here. The algorithm will combine the results to enhance the separation of fault types.

(**Work in progress**)

## Find out which process variables relate the most to each discovered fault type:
After separating the faulty observations into clusters, each cluster’s temporal residual data is analyzed to determine the variables most associated with the underlying fault type. The temporal residual data is used for this analysis because it isolates the effect of the fault on the process by removing the nominal process behavior from the observed data. This approach ensures that the identified variables are directly linked to the faults, providing a clearer understanding of the fault mechanisms.
#### Brief description of the fault localization method
The process begins by applying Principal Component Analysis (PCA) to the cluster, which involves obtaining the singular value decomposition of the temporal residual data associated with the cluster to obtain the loading vectors. Afterward, the loading vectors that capture the majority of the variance are identified. Typically, the first one or two loading vectors and their corresponding singular values are stored. Next, a feature relevance score is calculated, taking into account the importance of the loading vectors and the correlation of the process variables to these vectors. The process variables are then ranked in descending order based on their relevance scores. Finally, the top variables are reported for fault localization, with this work specifying that five variables are to be reported.
#### Fault localization results
Part of the results is shown in the table below



### Results Table
| Fault Type | PCA | Biased Variables |
|------------|-----|------------------|
| $f_2$ | $x_{34},x_{10},x_{47},x_{28},x_{30}$ | $x_{10},x_{47}, x_{22}, x_{17}, x_6$ |
| $f_4$ | $x_9,x_{45}, x_{21}, x_{22}, x_{43}$ | $x_{51}, x_9$ |
| $f_6$ | $x_{51},x_{46}, x_{52}, x_{45}, x_{42}$ | $x_{51}.x_{46}, x_{45},x_{52}, x_{21}$ |
| $f_{11}$ | $x_{51}, x_9, x_{31}, x_{32}, x_{38}$ | $x_{51}$ |
| $f_{12}$ | $x_{29},x_{23},x_{22},x_{25},x_{26}$ | |
| $f_{14}$ | $x_9,x_{51},x_{5},x_{21},x_4$ | |
| $f_{21}$ | $x_8,x_{21},x_{25},x_{26},x_3$ | |

### Results Ground Truth
![ground_truth](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/fault_loc_groundtruth.png)






