# LLDL_Kik-net

Accurate prediction of soil seismic response is necessary for geotechnical engineering. The conventional physics-based models such as the finite element method (FEM) usually fail to obtain accurate predictions due to the model assumption and parameter uncertainties. And the physics-based models are computationally expensive. This study proposed deep learning models to develop data-driven surrogate models for the prediction of soil seismic response based on the recorded ground motions from KiK-net downhole array sites. Two kinds of advanced neural networks, convolution neural network (CNN) and long short-term memory (LSTM) neural network, are applied in this framework respectively. These models do not rely on any prior knowledge about the soil site. The performance of the deep learning models is demonstrated through both numerical and recorded examples. Compared with the state-of-art FEM models, the proposed models could achieve better prediction performance with higher efficiency. The average prediction error is reduced by more than 40% in time domain and 30% in frequency domain. Even though great variability exists during the propagation of seismic in the reality, the models can still get satisfactory predictions.

For more information, please refer to the following:
* Li, L., Jin, F., Huang, D., Wang, G., 2023. [Soil seismic response modeling of KiK-net downhole array sites with CNN and LSTM networks](https://doi.org/10.1016/j.engappai.2023.105990). Engineering Applications of Artificial Intelligence 121, 105990

## Citation
<pre>
@article{LI2023105990,
title = {Soil seismic response modeling of KiK-net downhole array sites with CNN and LSTM networks},
journal = {Engineering Applications of Artificial Intelligence},
volume = {121},
pages = {105990},
year = {2023},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2023.105990},
url = {https://www.sciencedirect.com/science/article/pii/S0952197623001744},
author = {Lin Li and Feng Jin and Duruo Huang and Gang Wang},
keywords = {Deep learning, CNN, LSTM, Soil seismic response, Time series prediction}
}
</pre>
