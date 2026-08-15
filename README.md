
# Policy Gradient Methods for TVC

### Introduction



The **goal** of this project was to use policy gradient methods with deep neural networks to obtain stable flight for a TVC model rocket. In particular I used the popular policy gradient method **REINFORCE**, with a **feed-forward neural network** with 4 inputs, 4 outputs, and 2 hidden layers, each containing 64 hidden units. The success criterion was defined as maintaining both the x and y axis rotation of the model rocket within 5 degrees for the entire duration of the rocket’s burn. This criterion was successfully achieved.

### Problem Overview
In general a model rocket is said to be stable if the center of pressure(which is the point where all the aerodynamics forces seem to concentrate) is behind the center of gravity, since the torques generated from the forces at the CP would restore the model rockets orientation, allowing it fly stable. This can be seen using the right hand rule and using some imagination of scenarios. 

Due to some aerodynamics, model rockets with **fins** typically get the CP behind the CG. But if you have a model rocket with no fins then typically the CP wont be behind the CG, and thus you want have that aerodynamic restoring force and the model rocket won't fly stable. 

The model rocket I am building will not have any fins, so I must rely upon some other method for keeping the rocket stable in flight. The method I will use is Thrust Vector Control(TVC), which like the name implies, it controls the thrust vector. The way this is done is by mounting the motor to a gimbal, so that the thrust vector can be moved around to desired angles to keep the model rocket stable. 

The question is then which angles must you move the thrust vector to, such that the model rocket remains stable through its flight. Enter REINFORCE. 

## MDP Formulation
 
 #### Environment
 The environment the agent interacts with is the 6DOF physics simulator. The sim uses numerical integration for getting position and rotations via Newtons second law, uses quaternions for rotations, simulates wind with stochastic turbulence models, uses servos slewing and partial misalignment, and uses basic mass depletion for burned fuel. The coordinate system the simulator uses a typical right hand one with, 
$$
{+\hat z = \hat x \times \hat y}
$$

The simulator is quite "bare bones" due to the fact this was really meant to test policy gradient methods and not for flying actual rockets and the simulator was built around a ~1kg model rocket flying a Estes F-15-0 motor.  

This is an episodic task, where each powered flight is an episode.

#### State Space
A state needs to represent some data about the rockets orientation since that's what concerned about in the problem. So I decided to represent a state as a 4 tuple, consisting of the rotation of the rocket about the x and the y, as well as the angular velocity about the x and the y. Note do not care about the third degree of freedom, z, since TVC has no control over that. Each state is then given by
$$
{\vec s =(\theta, \phi, \omega_x, \omega_y)^T}
$$
Where ${\theta}$ and ${\phi}$ are the rotations about the x and y respectively and ${\omega_i}$ is the corresponding angular velocity. This mean that the state space is continuous leading to having to use non-tabular reinforcement learning methods to solve this problem.

#### Action Space
The agent outputs angle commands for the two gimbal axes. The particular gimbal designed has a range of motion of ${[-5^\circ, 5^\circ]}$, per axis, defining the action space as
$$
{a_i = [-5^\circ, 5^\circ], \text{  } i\in\{x, y\} }
$$

Thus the action space is also continuous, leading to the use of policy gradient methods to solve this problem.  

#### Reward 
The reward model in RL is something very important, since this signal is that essentially gets the agent to accomplish the goal. Since I want the model rocket to be stable, meaning ideally have angles of zero on rotation about x and y, then the reward model must shaped around that. So the reward model I picked was,
$$
r_t(\theta, \phi) = 
\begin{cases} 
-C & , |\theta| > 0.34 \lor |\phi| > 0.34 \\ 
e^{-a(\theta^2+\phi^2)} & ,\text{otherwise} 
\end{cases}
$$
Where ${C}$ and ${a}$ are hyperparameters, and if agent gets reward of ${C}$ the episode is terminated. 

## Policy