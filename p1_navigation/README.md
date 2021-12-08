[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

## Install instructions 

Install pytorch from conda

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

then install the requierement file:

`pip install requirements.txt`

## How to run each environment?

Basically you need to set each configuration for each algorithm, these configurations need to be changes into the `conf` folder,
the structure of that folder is the folowing:

* [`agent.yaml`](./conf/agent.yaml): for all the configurations of the Q-Learning agent (important one if you want to set dueling-dqn or double-dqn)
* [`config.yaml`](./conf/config.yaml): to set the seed and the train device for pytorch (no modification needed here)
* [`model.yaml`](./conf/model.yaml): hyperparameters for the DNN algorithm
* [`replay.yaml`](./conf/replay.yaml): all the hyperparameters for the (prioritized) experience replay.
* [`trainer.yaml`](./conf/trainer.yaml): hyperparameters for the trainer algorithm (episodes, actions per episode, epsilon, etc.)

## how to run the environment with a pretrained model?

Go to the [`Navigation.ipynb`](./Navigation.ipynb), to the 4th section **It's your turn!**, run the two subsections **Set the environmnet for python** and **Import all dependencies**. Then, go to the 
5th section **Test new agenn over the environment** and run the two cells below to run the pretrained agent.

That configuration is the final config (Double-DQN and Dueling-DQN) running, if you want to run new configurations, then:

#### **Base Q-Network**

Enter to the [`agent.yaml`](./conf/agent.yaml) and set the variables as follows:

* `network_kind`: `qnetwork`
* `loss_kind`: `dqn`

in the [`Navigation.ipynb`](./Navigation.ipynb) set the variable `weights_file` to `"./final_weights/checkpoint_all.pth"`, reset the notebook and run the cells as described before

#### **Double DQN**

Enter to the [`agent.yaml`](./conf/agent.yaml) and set the variables as follows:

* `network_kind`: `qnetwork`
* `loss_kind`: `ddqn`

in the [`Navigation.ipynb`](./Navigation.ipynb) set the variable `weights_file` to `"./final_weights/checkpoint_ddqn.pth"`, reset the notebook and run the cells as described before.

#### **Dueling DQN**

Enter to the [`agent.yaml`](./conf/agent.yaml) and set the variables as follows:

* `network_kind`: `dueling_qnetwork`
* `loss_kind`: `dqn`

in the [`Navigation.ipynb`](./Navigation.ipynb) set the variable `weights_file` to `"./final_weights/checkpoint_dueling.pth"`, reset the notebook and run the cells as described before.

#### **Prioritized Experience Replay**

Enter to the [`agent.yaml`](./conf/agent.yaml) and set the variables as follows:

* `network_kind`: `qnetwork`
* `loss_kind`: `dqn`

then go to [`replay.yaml`](./conf/replay.yaml) and set `alpha` to `0.6`, (**NOTE** set it again to `0.0` when finished for other experiments)

in the [`Navigation.ipynb`](./Navigation.ipynb) set the variable `weights_file` to `"./final_weights/checkpoint_per.pth"`, reset the notebook and run the cells as described before.

#### **Dueling-DQN + Double-DQN**

Enter to the [`agent.yaml`](./conf/agent.yaml) and set the variables as follows:

* `network_kind`: `dueling_qnetwork`
* `loss_kind`: `ddqn`

in the [`Navigation.ipynb`](./Navigation.ipynb) set the variable `weights_file` to `"./final_weights/checkpoint_all.pth"`, reset the notebook and run the cells as described before.

this is the environment set by default in the repository.

## How to run the environment from scratch? 

Run the cells of the all 4th section with the configuration that you may want, at the end it should generate a weights file `./final_weights/checkpoint_final.pth`,
set the variable `weights_file` to that new path and run all the cells of the 5th section.

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
