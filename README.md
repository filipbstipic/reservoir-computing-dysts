# Reservoir Computing for Chaotic Dynamical Systems

This project is the continuation of the group project done for the [Deep Learning Course at ETH Zurich](http://da.inf.ethz.ch/teaching/2021/DeepLearning/) in Fall '21. It has received a grade of 5.775/6.0, whereas a grade of 6.0 by ETH Zurich standards implies "Good enough for submission to an international conference". The original group project repository can be found [here](https://github.com/FatjonZOGAJ/dysts/tree/reservoir_computing).

For more information, please check the [report](report.pdf).

__Abstract:__ Chaotic dynamical systems continue to puzzle and amaze practitioners due to their inherent unpredictability, despite their finite and concise representations. In spite of its simplicity, Reservoir Computing ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) has been demonstrated to be well-equipped at the task of predicting the trajectories of chaotic systems where more intricate and computationally intensive Deep Learning methods have failed, but it has so far only been evaluated on a small and selected set of chaotic systems ([P.R. Vlachas et al.](https://doi.org/10.1016/j.neunet.2020.02.016)). We build and evaluate the performance of a Reservoir Computing model known as the Echo State Network ([H. Jaeger et al.](https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note')) on a large collection of chaotic systems recently published by [W. Gilpin](https://arxiv.org/abs/2110.05266) and show that ESN does in fact beat all but the top approach out of the 16 forecasting baselines reported by the author.



## Main installation

    virtualenv rc --python=python3.7
    source rc/bin/activate
    pip install -r requirements.txt

# Code
We have adapted the code from the https://github.com/williamgilpin/dysts repository which provides the initial implementations of the dynamical chaos systems as well as the original benchmark. 
For reproducibility reasons we have tried to keep the existing code and experiments as similar as possible and have e.g. *not* refactored existing redundant code as it could potentially  change results and introduce errors.
This additionally will allow for easier pull requests when we push our changes to the original repository. For that reason we ask for leniency regarding code quality.

# Rerunning Experiments

To get solid performance run:

python main.py --reservoir_size=1000  --radius=0.9 --sparsity=0.1 --alpha=1.0 --reg=1e-7 --seed=10

