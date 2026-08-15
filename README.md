# Policy Gradient Methods for TVC 

### Introduction
The **goal** of this project was to use policy gradient methods with deep neural networks to obtain stable flight for a TVC model rocket. In particular I used the popular policy gradient method **REINFORCE**, with a **feed-forward neural network** with 4 inputs, 4 outputs, and 2 hidden layers, each containing 64 hidden units. The success criterion was defined as maintaining both the x and y axis rotation of the model rocket within 5 degrees for the entire duration of the rocket’s burn. This criterion was successfully achieved.