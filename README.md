
# Unsupervised Fault Discovery 

Industrial systems require efficient and rapid corrective actions when faults occur. A fault is an abnormality that causes deviation from an expected behavior. When not resolved in a timely manner, faults can damage the equipment and lead to a loss of money and time. For example, a broken temperature sensor can indicate that a component's temperature is normal when it is off the chart! Sometimes, the effect of fault might be catastrophic, such as in the case of the [Fukushima](https://en.wikipedia.org/wiki/Fukushima_nuclear_accident) and [Deepwater Horizon](https://en.wikipedia.org/wiki/Deepwater_Horizon) incidents.

<p align="center">
  <img src="https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/Fukushima%20Incident.jpg" alt="Image Fuku" width="45%"/>
  <img src="https://github.com/mbakr99/unsupervised-fault-discovery/blob/main/imgs/Deepwater%20Horizon%20Explosion.jpg" alt="Image Deep" width="45%"/>
</p>

Fault diagnosis is concerned with addressing the faults that a system might encounter by consistently monitoring the process and generating alarms and relative information to the operators. 
With the evolution of sensor technology, data storage, and computational power. Data-driven fault diagnosis methods have started to replace classical methods, which tend to be impractical when the industrial process of interest is complex. Data-driven methods rely **solely** on historical process data to learn how to make appropriate decisions. This can include but is not limited to, detecting when a fault is present, determining the source of the fault, or classifying the fault categories. The best of these is mapping the fault into its corresponding type, as this enables the operator to know everything about the fault before taking action. In fact, once the type is known, the operator can follow a predetermined procedure set by the field expert to address the fault optimally.
