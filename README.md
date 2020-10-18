#<div align="center">

# The Plastic Support Structure

**A network augmentation that allows for efficient multi-task training.**
</div>

## Abstract
We propose a novel approach to lifelong learning, introducing a compact encapsulated support structure which endows a network with the capability to expand its capacity as needed to learn new tasks while preventing the loss of learned tasks. This is achieved by splitting neurons with high semantic drift and constructing an adjacent network to encode the new tasks at hand. We call this the Plastic Support Structure (PSS), it is a compact structure to learn new tasks that cannot be efficiently encoded in the existing structure of the network. We validate the PSS on public datasets against existing lifelong learning architectures, showing it performs similarly to them but without prior knowledge of the task and in some cases with fewer parameters and in a more understandable fashion where the PSS is an encapsulated container for specific features related to specific tasks, thus making it an ideal "add-on" solution for endowing a network to learn more tasks.

## Detailed High Level Process
When a neuron has high semantic drift after training a new task, duplicate it and revert it to it's old weight value. 
For each layer in which a neuron has not been split, add a new neuron.
Connect all new neurons together, and in the last hidden layer to the appropriate output neuron.
Set the outgoing weights of new neurons to old neurons to zero. Train these new neurons exclusively.


## Software 
# How to Run
1. Install the dependencies below.
2. Pick the experiment you'd like to run, find it's corresponding run_experiment.py file.
3. Run prepare_experiment()
4. Optionally, run find_hypers() to get a good set of hyperparameters.
5. Run train_model(). Parameters are defined inside the function.


# Dependencies
- Python 3.7
- numpy 1.18.4
- Pillow 7.0.0 or Pillow-SIMD 7.0.0
    - For pillow-simd you will need a SSE4.1 capable CPU and `sudo apt-get install libjpeg-dev zlib1g-dev`
- progress 1.5
- torch 1.5.0
- torchvision 0.6.0
- sklearn 0.0
- matplotlib 3.2.1
- Ray[Tune] 0.8.5 (`pip install 'ray[tune]'`)

## Style Guide
We use the industry standard [PEP8].

[PEP8]: <https://pep8.org>

## Resources
Below a compilation of resources we've used along the way, including academic papers, tutorials, etc.
### Papers
- [Lifelong Learning with Dynamically Expandable Networks](https://openreview.net/pdf?id=Sk7KsfW0-) with [code](https://github.com/jaehong-yoon93/DEN)
### Datasets
- MNIST
- MNIST Variations

## Thanks
Thanks to Prof. John Vervaeke for guidance, thoughts and wisdom.

Thanks to Prof. Steve Mann for providing some computational resources.

Thanks to bjsowa/DEN for inital fork structure.

Thanks to mikacho for guidance on the resource constraining function.


## Citation
Submitted to JMLR.
