---
hide-toc: true
firstpage:
lastpage:
---

# NetworkGym: Democratizing Network AI
NetworkGym is an innovative Simulation-as-a-Service framework designed to democratize Network AI Research and Development. At the core of NetworkGym is the streamlined concept of the "agent-environment loop" in reinforcement learning.

```{figure} network_gym_intro.png
---
width: 100%
---
```

## Benefits of NetworkGym
NetworkGym is partitioned into Open-Source (Agent/Client) and Service (Environment) components, yielding the following benefits:
- **Agent Training without Needing Network Simulation Expertise**: The Agent is tailored for open-source utilization, whereas the Environment is under our management. By utilizing open-source NetworkGym Client and API, the Agent interacts with the Environment through public internet connections, enabling agent training without the requirement of network simulation expertise.
- **Flexibility in Language and Platform**: The separation of the Agent and Environment provides the freedom to employ various programming languages. For instance, a Python-based Agent can smoothly interact with a C++(ns-3) based simulation Environment. 
- **Controlled Access through the NetworkGym API**: By leveraging the networkgym API for message exchanges, users are prevented from direct access to the simulator or the underlying network environment implementation. This controlled access maintains security and oversight.
- **Independent Deployment**: Separating the Agent and Environment allows them to be deployed on distinct machines or platforms, optimized for specific workloads. This approach also accommodates the scenario where they can be developed and maintained by different entities.


## Agent
NetworkGym offers four types of network agents, each serving a specific purpose when interacting with NetworkGym Environments:
- **System Default Agent**: By sending an "empty" action to the environment, this agent enables the collection of data for offline training purposes.
- **Custom Algorithm Agent**: A flexible option that allows users to define their own specialized agents.
- **Stable-Baselines3 Agent**: A RL agent includes the State-of-the-Art (SOTA) Reinforcement Learning (RL) algorithms sourced from Stable-Baselines3.
- **CleanRL Agent**: A CleanRL agent is available for custom algorithms.

We have included additional information on how to train the agent in the [Training Agents](tutorials/training_agents.md) tutorial. This tutorial provides detailed guidance and instructions for effectively training your agent using various network agents offered by NetworkGym.


## Gymnasium API
The interaction between the network agent and the environment is facilitated through the standard [Gymnasium](https://gymnasium.farama.org/) API, including action (↓), observation (↑), and reward (↑).

## NetworkGym API
NetworkGym consists of three essential components and two interfaces, each playing a crucial role in the framework:
- **NetworkGym Client**: connects Agent to the Server, configures the Environment, and enables data conversion between Gym and NetworkGym format.
    - **Northbound Interface**: interface between Client and Server, including network configuration (↓), action (↓), and network stats (↑).
- **NetworkGym Server**: manages connections between Clients and Environments.
    - **Southbound Interface**: interface between Server and Environment, including network configuration (↓), action (↓), and network stats (↑).
- **NetworkGym Environment**: simulates or implements network environments tailored for various network AI use cases. E.g., open-source network simulation tools such as [ns-3](https://www.nsnam.org/).

For more comprehensive information about the key components of NetworkGym, please refer to the [Overview](content/overview.md) page. 

## NetworkGym Scope and Limitations

✔️ **In-Scope:** 
- NetworkGym focuses on the development of open and reference AI models/algorithms tailored for networking research (Layer 2 and above). 
- It also involves the development of abstract models, such as channel, compute, power, etc., simulating aspects of the physical world and systems.
- NetworkGym endeavors to create full-stack models that can effectively simulate end-to-end (Access/Edge/Cloud) network operations with high fidelity.

❌ **Out-of-Scope:**
- NetworkGym does not involve the development of proprietary AI algorithms or models. 
- It is not centered around creating AI applications or software, like xApp/rApp, based on specific controller architectures (e.g., O-RAN RIC).
- NetworkGym is not designed to create a Digital Twin to simulate the physical world with high fidelity.

```{toctree}
:hidden:
:caption: Introduction

content/motivation
content/overview
content/quickstart
```

```{toctree}
:hidden:
:caption: API

api/network_gym_client
api/network_gym_server
api/network_gym_env
```

```{toctree}
:hidden:
:caption: Environments

environments/mx_traffic_management
environments/mx_network_slicing
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/handling_time_limits
tutorials/training_agents
tutorials/customizing_observation_and_reward
tutorials/implementing_custom_environment
Gymnasium Tutorial <https://gymnasium.farama.org/tutorials/gymnasium_basics/>

```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/IntelLabs/networkgym>
contact
```