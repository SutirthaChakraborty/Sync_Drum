
# Virtual Gesture-Based Drum Synchronizer 


## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Code Structure](#code-structure)
4. [Methodology](#methodology)
5. [Features and Advantages](#features-and-advantages)
6. [Future Work](#future-work)

<a name="description"></a>
## Description

"Virtual Gesture-Based Drum Synchronizer" is an innovative application that harnesses the power of complex system dynamics, allowing users to play virtual drums using gestures, all while staying perfectly in sync with any ongoing rhythm or music. Utilizing state-of-the-art algorithms, such as `Swarmlator`, `Janus`, and `Kuramoto`, this project decodes and predicts user gestures to create real-time drumming experiences.

Imagine air drumming, but every beat you hit is perfectly recognized and reproduced in harmony with the song you're listening to. Whether you're a professional musician looking for a futuristic drumming experience or just someone who loves to groove to the beat, the "Virtual Gesture-Based Drum Synchronizer" transforms your gestures into rhythmic symphonies. Experience drumming like never before - no drums needed, just your passion for rhythm!

 📁 Directory Structure
│
├─ 🎶 a.wav - Sample song file
├─ 🥁 drum-a.wav - Drum kit sample A
├─ 🥁 drum-b.wav - Drum kit sample B
│
├─ 🐍 Janus.py - Janus implementation
├─ 🐍 swamlator.py - Swarmlator implementation
├─ 🐍 kuramoto.py - Kuramoto implementation
├─ 🐍 Main.py - Main file to run
├─ 🐍 simulation.py - Contains required classes for implementation
│
├─ 📄 sonicpi.txt - Code for Sonic Pi
├─ 🌱 environment.yml - Conda environment details
└─ 🔧 requirements.txt - Pip requirements

<a name="installation"></a>
## Installation

To get started with the simulations, follow these installation steps:

1. Clone the repository:
   ```
   git clone https://github.com/SutirthaChakraborty/Sync_Drum
   ```

2. Navigate to the cloned directory:
   ```
   cd Sync_Drum
   ```

**Requirements**:
### 1. Setting up the Conda environment

  

To set up the conda environment, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, navigate to the project directory and run:

  

```bash
conda  env  create  -f  environment.yml
conda  activate  Kuramoto
```

  

Replace `Kuramoto` with the name you've given in the `environment.yml` file.

  

### 2. Installing requirements via pip

  

If you prefer `pip`, you can install the necessary packages using:

  

```bash
pip  install  -r  requirements.txt
```

  

### 3. Running the Project

  

Navigate to the project directory and run:

  

```bash
python  main.py
``` 

 
<a name="code-structure"></a>
## Code Structure

- **main.py**: Entry point for running the simulations.
- **simulation.py**: Base class for simulations. It includes basic functionality like input/output modes, plotting, and other utility functions.
- **janus.py**: Contains the `Janus` class that simulates interactions between two systems.
- **swarmlator.py**: Contains the `Swarmalator` class that delves into the complex interactions seen in certain swarm-like dynamics.
- **kuramoto.py**:  This file  typically model the Kuramoto oscillators, a paradigmatic example in the study of synchronizations in complex systems.

<a name="methodology"></a>
## Methodology

1. **Simulation Base Class**: At the heart of our simulation lies the `Simulation` base class that sets the groundwork for advanced simulations. This class takes care of essential tasks such as setting up input/output modes, handling live data, plotting results, and more.

2. **Janus**: The Janus class is named after the two-faced Roman god, representing its capability to model two interplaying dynamics. The simulation takes care of scenarios where two systems interact in a complex manner.

3. **Swarmalator**: This is a unique combination of swarm dynamics (where entities move in cohesion) and oscillators (entities with periodic behavior). The class aims to simulate situations where entities display both swarm-like and oscillatory behaviors.

4. **Kuramoto Oscillators**: This renowned model, assumed to be in `kuramoto.py`, describes a large system of coupled oscillators and has been pivotal in understanding synchronization phenomena.

<a name="features-and-advantages"></a>
## Features and Advantages

- **Versatility**: With three distinct models on offer, this suite caters to a wide range of complex system dynamics.
- **Live Data Handling**: Especially with the Swarmlator and Janus, live data can be fed, making it relevant for real-time applications.
- **Visualization**: Built-in functionalities to visualize the dynamics offer intuitive insights into system behaviors.
- **Expandability**: The modular structure of the code allows for easy expansions and inclusion of more complex systems in the future.

<a name="future-work"></a>
## Future Work

- **Incorporation of Machine Learning**: Future versions could incorporate machine learning to predict system behaviors based on past data.
- **Multi-Dimensional Analysis**: Expanding the models to multi-dimensional spaces to simulate more realistic scenarios.
- **Optimization**: Leveraging faster algorithms and parallel processing to handle larger datasets in real-time.
- **UI/UX Improvements**: Building a user-friendly interface to interact with the simulations seamlessly.

---

For feedback, contributions, or questions, please reach out to [sutirtha.chakraborty@mu.ie].