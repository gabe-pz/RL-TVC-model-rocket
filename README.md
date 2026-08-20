# Policy Gradient Methods for TVC

### Introduction

The **goal** of this project was to use policy gradient methods with deep neural networks to obtain stable flight for a TVC model rocket. In particular I used the popular policy gradient method **REINFORCE**, with a **feed-forward neural network** with 4 inputs, 4 outputs, and 2 hidden layers, each containing 64 hidden units. The success criterion was defined as maintaining both the x and y axis rotation of the model rocket within 5 degrees for the entire duration of the rocket’s burn. This criterion was successfully achieved.

  

### Problem Overview

In general a model rocket is said to be stable if the center of pressure(which is the point where all the aerodynamics forces seem to concentrate) is behind the center of gravity, since the torques generated from the forces at the CP would restore the model rockets orientation, allowing it fly stable. This can be seen using the right hand rule and using some imagination of scenarios.


Due to some aerodynamics, model rockets with **fins** typically get the CP behind the CG. But if you have a model rocket with no fins then typically the CP wont be behind the CG, and thus you wont have that aerodynamic restoring force and the model rocket won't fly stable.

  
The model rocket I am building will not have any fins, so I must rely upon some other method for keeping the rocket stable in flight. The method I will use is Thrust Vector Control(TVC), which like the name implies, it controls the thrust vector. The way this is done is by mounting the motor to a gimbal, so that the thrust vector can be moved around to desired angles to keep the model rocket stable.

  
The question is then which angles must you move the thrust vector to, such that the model rocket remains stable through its flight. Enter REINFORCE.

  
## MDP Formulation

#### Environment

The environment the agent interacts with is the 6DOF physics simulator. The sim uses numerical integration for getting position and rotations via Newtons second law, uses quaternions for rotations, simulates wind with stochastic turbulence models, uses servos slewing and partial misalignment, and uses basic mass depletion for burned fuel. The coordinate system the simulator uses a typical right hand one with,

$$
{+\hat z = \hat x \times  \hat y}
$$


The simulator was built around a model rocket with a mass of about 1kg and flying an Estes F-15-0 motor. Thus the aerodynamics are modeled for rockets flying well under supersonic speeds.

Also this is an episodic task, where each powered flight is a single episode. 
  

#### State Space

A state needs to represent some data about the rockets orientation since that's what concerned about in the problem. So I decided to represent a state as a 4 tuple, consisting of the rotation of the rocket about the x and the y, as well as the angular velocity about the x and the y. Note do not care about the third degree of freedom, z, since TVC has no control over that. Each state is then given by

$$
{\vec s =(\psi, \phi, \omega_x, \omega_y)^T}
$$

Where ${\psi}$ and ${\phi}$ are the rotations about the x and y respectively and ${\omega_i}$ is the corresponding angular velocity. This mean that the state space is continuous leading to having to use non-tabular reinforcement learning methods to solve this problem.

  

#### Action Space

The agent outputs angle commands for the two gimbal axes. The particular gimbal designed has a range of motion of ${[-5^\circ, 5^\circ]}$, per axis, defining the action space as

$$
{a_i = [-5^\circ, 5^\circ], \text{ } i\in\{x, y\} }
$$

  

Thus the action space is also continuous, leading to the use of policy gradient methods to solve this problem.

  

#### Reward

The reward model in RL is something very important, since this signal is that essentially gets the agent to accomplish the goal. Since I want the model rocket to be stable, meaning ideally have angles of zero on rotation about x and y, then the reward model must shaped around that. So the reward model I picked was,

$$
r_t(\psi, \phi) =
\begin{cases}
-c & , |\psi| > 0.34  \lor |\phi| > 0.34 \\
e^{-a(\psi^2+\phi^2)} - b(\dot{\psi}^2+\dot{\phi}^2) & ,\text{otherwise}
\end{cases}
$$

Where ${a}$, ${b}$, and ${c}$, are hyperparameters, and if agent gets reward of ${c}$ the episode is terminated.

  

## The Policy

As previously stated, the policy gradient method used here is *REINFORCE*. The policy is therefore stochastic and is parameterized by, ${\vec\theta}$, giving the policy of ${\pi(a_x,a_y|s,\vec\theta)}$. In general the **goal** of all policy gradient methods is to find the parameters of a policy that maximize the expected return. REINFORCE accomplishes this using a Monte Carlo estimate of the policy gradient. In particular, the parameters are updated after an episode has been completed, using the return observed from the sampled trajectory. This means that the complete trajectory must be collected before an update can be performed, i.e updated on an episodic basis. The pseudocode for the REINFORCE algorithm given by Sutton and Barto, which was used as the basis for the implementation here, is shown below.

![enter image description here](https://cdn.phototourl.com/free/2026-08-15-1c0780be-3111-4082-9854-a9df1689025a.png)

  

#### Stochastic Policy Formulation

Since the action space is continuous, the policy must be defined using a probability distribution that can represent continuous actions. For this problem, I chose the univariate normal distribution, aka the Gaussian distribution, to model each action component. Assuming that each action component is independent, then can write the policy as such

$$
{\pi(a_x,a_y|s,\vec\theta) = \pi(a_x|s,\vec\theta)\pi(a_y|s,\vec\theta)=\mathcal{N}(\mu_x, \sigma_x^2)\mathcal{N}(\mu_y, \sigma_y^2)}
$$


The goal here is to then find the parameters of each probability distribution, ${\mu_i}$ & ${\sigma_i}$, such that obtain an optimal policy. This was done using a feed-forward neural network whose parameters were given by ${\vec\theta}$. That is letting the outputs of the network be ${\vec\kappa}$ then

$$
{\vec\kappa = f[\vec s,\vec\theta]=(\mu_x, \log(\sigma_x), \mu_y, \log(\sigma_y)})^T
$$

Where use logarithms in outputs of network because ${\sigma}$ must be positive, so have the network output the ${\log\sigma}$, then exponentiation the output to ensure it is positive.

  

#### Action selection

During training to select an action it is done simply by sampling an action from its corresponding distribution(x or y), then due to the constraints of the gimbal those "raw actions" are passed into hyperbolic tangent and multiplied by the max output angle to ensure actions are within physical limits.

  

After training then want to select the action with the highest probability each time, which for the Gaussian, just means selecting ${\mu}$ as the raw action and follow same action "squashing" process


## The Network
As stated before the neural network architecture I used was a feed-forward multilayer perceptron (MLP). It took the state vector ${\vec s}$ and outputted parameters for each independent policy, ${\pi_x}$ and ${\pi_y}$, as such,

$$
{\vec\kappa = f[\vec s, \vec\theta]=(\mu_x, \log(\sigma_x), \mu_y, \log(\sigma_y)})^T
$$

The MLP I used consists of 2 hidden layers, each with 64 hidden units. 

The activation function I used was the rectified linear unit, or ReLU for short. I only applied it to the 2 hidden layers and not the output layer because ReLU is defined as

$$
{\text{ReLU}(z) =\begin{cases}
0 & ,z <0 \\
z & , z\geq 0
\end{cases} }
$$

meaning its range is ${(0,+\infty)}$. But for the Gaussian, ${\mu}$ is defined on ${(-\infty, \infty)}$, so in the laziest way to account for that, I simply removed the activation on the output layer and kept it linear.

## Training 
All that was used to train the network was vanilla stochastic gradient ascent with gradient clipping. After 50,000 episodes the policy was able to converge to achieve the success criterion. The hyperparameters used are 
given here

```cpp
//hyperparameters
float alpha = 0.00005f;//step size
float gamma = 0.8f;//discount factor 
float a = 175.0f;//exp constant
float b = 5.5f;//angular v penalize factor
int c = -10;//termination reward

float maxStep = 1.0;//max step for grad clipping
```
A plot of episodes vs average return, logged every 1k episodes, is given below, 

![enter image description here](https://cdn.phototourl.com/free/2026-08-17-b803d83b-2242-4c7c-bd28-d7187ddd827b.png)


## Results
The results of the flight after 50,000 episodes of training and a burn time of ${t=3.45s}$ are given below in the plots showing successfully achievement of the criterion,


### Rotation about the x-axis for the flight

![enter image description here](https://cdn.phototourl.com/free/2026-08-16-e361be0d-fb84-4f8f-a164-b8a86d978b9a.png)

### Rotation about the y-axis for the flight

![enter image description here](https://cdn.phototourl.com/free/2026-08-16-f388cba3-3495-4664-bae7-88d1d20de1c1.png)
