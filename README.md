
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
-  Find out which process variables relate the most to each discovered fault type

The proposed work is tested on the Tennessee Eastman Process (TEP). The full analysis is provided in this [Jupyter notebook](https://github.com/mbakr99/unsupervised-fault-discovery/blob/e5830f0c26041d850854d53189ec85d23f97779b/FCT_6.ipynb). This file provides a high-level description of the method. It also provides a discussion and interpretation of the results. 

## Detect Faults:
The proposed work relies on capturing the behavior of the process during NOC. This is done using two probabilistic models that capture two aspects of the NOC behavior. First, a temporal probabilistic model is used to capture the temporal relation between consecutive process measurements. This model reports a fault when the NOC temporal behavior is violated. The second is a static distribution to capture the correlation and the range information of the process variables during NOC. This model is concerned with how a single observation fits the trend learned from the NOC data. It reports a fault when the NOC static characteristics are violated. 
### Temporal model:
A Probabilistic Recurrent Neural Network (PRNN) of a custom structure was created using [Tensorflow](https://www.tensorflow.org) and [Tensorflow probaility](https://www.tensorflow.org/probability)  and used to capture the NOC temporal behavior. The modelt utilizes the latest measurement to predict the distribution of the future one. The actual measurement is then compared to the distribution, and a fault is reported based on "how well it conforms to the distribution". The NOC data was partitioned into training and testing sets. The training set was normalized using standard normalization, and the normalization parameters were saved to normalize the model inputs at testing and deployment. More details can be found in this [FCT_6.ipynb](https://github.com/mbakr99/unsupervised-fault-discovery/blob/e5830f0c26041d850854d53189ec85d23f97779b/FCT_6.ipynb). The prediction of the model on the testing set vs the actual measurements is shown below 

![Figure RNN results](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/pred_val.png)          

Note that the temporal distribution captures the NOC behavior accurately as the actual measurments fall well within the confidence interval of the prediction at each time step.
A description of the process variables that appear in the above figure can be found in this [source paper](https://doi.org/10.1016/0098-1354(93)80018-I). 

### Static model:
A multivariate gaussian distribution was used to capture the static charechterstics of the NOC data. Unlike the previous model, this model does not change with time. The following displays the model for the first three process variables. Notice how the actual NOC data distributin matches the model. 

![Figure joint](https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/joint_dist.png)

### Combining both models:
An anomlay score is a value that incdicates the likelihood of the 

