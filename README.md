# Reservoir Computing for Chaotic Dynamical Systems

This project is the correction, update and continuation of the group project done for the [Deep Learning Course at ETH Zurich](http://da.inf.ethz.ch/teaching/2021/DeepLearning/) in Fall '21. The original project had received a grade of 5.775/6.0, whereas a grade of 6.0 by ETH Zurich standards implies "Good enough for submission to an international conference"; it can be found [here](https://github.com/FatjonZOGAJ/dysts/tree/reservoir_computing). In this project experiments were re-run for a slighly more general version of the ESN, as well as new insight into the model performance is provided, alongside with updated plots/results. For more information, please check the [report](report.pdf).

__Abstract:__ Chaotic dynamical systems continue to puzzle and amaze practitioners due to their inherent unpredictability, despite their finite and concise representations. In spite of its simplicity, Reservoir Computing ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) has been demonstrated to be well-equipped at the task of predicting the trajectories of chaotic systems where more intricate and computationally intensive Deep Learning methods have failed, but it has so far only been evaluated on a small and selected set of chaotic systems ([P.R. Vlachas et al.](https://doi.org/10.1016/j.neunet.2020.02.016)). We build and evaluate the performance of a Reservoir Computing model known as the Echo State Network ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) on a large collection of chaotic systems recently published by [W. Gilpin](https://arxiv.org/abs/2110.05266) and show that ESN does in fact beat all but the top approach out of the 16 forecasting baselines reported by the author.


## Main installation

    virtualenv rc --python=python3.7
    source rc/bin/activate
    pip install -r requirements.txt


## Re-running Experiments

To run with best found hyperparameters, run:

    python main.py

To get solid performance for a single hyperparameter setting, run:

    python main.py --reservoir_size=1000  --radius=0.9 --sparsity=0.1 --alpha=1.0 --reg=1e-7 --seed=10



