# Frozen Lake Demo

This directory contains a comprehensive demonstration of the Frozen Lake environment from Gymnasium, showcasing various reinforcement learning concepts and algorithms with detailed training progress tracking and model persistence.

## What is Frozen Lake?

Frozen Lake is a classic reinforcement learning environment where:
- An agent starts at position S (start)
- The goal is to reach position G (goal) 
- The agent must avoid holes H in the ice
- F represents safe frozen surfaces
- Actions are: Left (0), Down (1), Right (2), Up (3)

## Project Structure

```
frozen_lake/
├── frozen_lake_demo.py      # Main demo with environment interaction
├── multi_algorithm_demo.py  # All RL algorithms (Q-Learning, SARSA, Expected-SARSA)
├── models/                  # Saved models and visualizations
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Key Features

### 📊 **Rich Visualizations & Training Tracking**
- Q-table heatmaps for each action direction
- Policy visualization with arrow directions  
- Training progress plots (rewards, epsilon, success rate, steps)
- Algorithm comparison plots with detailed metrics
- **Real training data**: Actual performance results with PNG visualizations
- Episode-by-episode progress tracking with success/failure status

### 🎯 **Detailed Training Progress**
- Episode-by-episode reward and status tracking
- Real-time training progress with success/failure status
- Comprehensive training history with metrics
- **Fast training**: No visual rendering during training for optimal speed

### 💾 **Model Persistence**
- Save trained models with timestamps and complete training history
- Load and reuse pre-trained models with performance metrics
- Model metadata includes hyperparameters and training statistics

### 🔄 **Modular Design**
- Multiple RL algorithms in unified framework (Q-Learning, SARSA, Expected-SARSA)
- Easy to extend with new algorithms
- Clean separation of concerns
- Comprehensive algorithm comparison tools

## Files Description

### `multi_algorithm_demo.py`
Central hub containing all reinforcement learning algorithms:
- **BaseRLAgent**: Common base class with shared functionality
- **QLearningAgent**: Q-learning implementation with detailed progress tracking
- **SarsaAgent**: SARSA (on-policy) algorithm implementation
- **ExpectedSarsaAgent**: Expected SARSA hybrid approach
- All algorithms include model save/load, evaluation metrics, and visualization methods
- Algorithm comparison and analysis tools

### `frozen_lake_demo.py`
Environment-specific demonstration script:
- Environment setup and visualization
- Random play demonstration
- Training workflows for any algorithm
- Interactive model loading options
- Policy and Q-table visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python frozen_lake_demo.py
```

Choose from comprehensive options:
1. 4x4 Grid (Slippery) - Standard challenging version
2. 4x4 Grid (Not Slippery) - Deterministic version
3. 8x8 Grid (Slippery) - Larger, more complex version
4. Compare All Configurations - Run all and compare
5. Load and test existing model - Use saved models
6. **Multi-algorithm comparison (All 11 algorithms)** - Complete comparison

### Training Options
Choose from **11 RL algorithms** plus utility options:

**Core Algorithms:** Q-Learning, SARSA, Expected-SARSA  
**Advanced Exploration:** Double-Q-Learning, UCB-Q-Learning, Boltzmann-Q-Learning  
**Enhanced Learning:** N-Step-Q-Learning, Monte-Carlo-Control, Dyna-Q  
**Experience Replay:** Experience-Replay, Prioritized-Replay  

### Multi-Algorithm Comparison
```bash
python multi_algorithm_demo.py
```
Trains all 11 algorithms with identical hyperparameters and generates comprehensive performance comparison with visualizations.

### Training Progress Display
Training shows detailed episode-by-episode progress without visual rendering for speed:

```
Episode    1: Reward=0.0, Steps= 4, Epsilon=0.999, Status=FAILED
Episode  500: Reward=1.0, Steps=14, Epsilon=0.606, Status=SUCCESS
Episode 1000: Reward=1.0, Steps= 6, Epsilon=0.367, Status=SUCCESS

Final Summary: Success Rate: 58.0%, Average Reward: 0.580
```

### Model Files & Loading

**Generated Files:**
- **Models (.pkl)**: Complete Q-tables, hyperparameters, training history for all 11 algorithms
- **Visualizations (.png)**: Training progress plots and algorithm comparisons

**Loading Example:**
```python
from multi_algorithm_demo import *

# Load any algorithm model
agent = load_any_model('models/prioritized_replay_model_20250619_122810.pkl')
print(f"Algorithm: {agent.algorithm_name}, Success Rate: {agent.training_history['success_rate']}")
```

### Direct Usage Example

```python
from multi_algorithm_demo import PrioritizedReplayAgent, compare_algorithms
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)

# Use top-performing algorithm
agent = PrioritizedReplayAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.95
)

# Train and evaluate
history = agent.train(env, episodes=1000)
results = agent.evaluate(env, episodes=100)
agent.save_model('my_model.pkl')

# Compare all algorithms
comparison_results, histories = compare_algorithms()
```

### Extending with New Algorithms

```python
class NewRLAgent(BaseRLAgent):
    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "New-Algorithm"
        # Add algorithm-specific initialization
    
    def update_q_value(self, state, action, reward, next_state, next_action=None, done=False):
        # Implement algorithm-specific update logic
        pass
```

The modular design automatically provides save/load, evaluation, and visualization methods.

## Performance Results & Algorithm Analysis

### Latest Training Results (4x4 Slippery Frozen Lake)

| Rank | Algorithm | Success Rate | Performance Tier | Notes |
|------|-----------|-------------|-----------------|--------|
| 🥇 | **Prioritized Experience Replay** | **63.0%** | **Tier 1** | Best overall performer |
| 🥈 | **Experience Replay Q-Learning** | **44.0%** | **Tier 1** | Strong sample efficiency |
| 🥉 | **SARSA** | **42.0%** | **Tier 1** | Consistent on-policy learning |
| 4 | **Boltzmann Q-Learning** | **29.0%** | **Tier 2** | Effective exploration strategy |
| 5 | **UCB Q-Learning** | **16.0%** | **Tier 2** | Principled exploration |
| 6 | **N-Step Q-Learning** | **16.0%** | **Tier 2** | Multi-step credit assignment |
| 7 | **Monte Carlo Control** | **9.0%** | **Tier 3** | Unbiased but high variance |
| 8 | **Q-Learning** | **7.0%** | **Tier 3** | Standard baseline |
| 9-11 | **Expected SARSA, Double Q-Learning, Dyna-Q** | **0.0%** | **Variable** | May need hyperparameter tuning |

### Performance Insights

**Experience Replay Methods Lead:** Prioritized Replay (63%) and Experience Replay (44%) demonstrate the power of intelligent experience reuse.

**On-Policy vs Off-Policy:** SARSA (42%) consistently outperforms Q-Learning (7%) in stochastic environments.

**Advanced Exploration:** Boltzmann (29%) > UCB (16%) > Epsilon-Greedy, showing temperature-based exploration advantages.

### Expected Performance Ranges

**4x4 Deterministic:** Most algorithms achieve 85-100% success rate  
**4x4 Slippery:** Performance ranges from 0-63% as shown above  
**8x8 Slippery:** Projected 10-40% success rates (more challenging)

## Algorithm Technical Deep Dive

### Core Learning Mechanisms

#### **Q-Learning (Off-Policy TD Learning)**
**Update:** `Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`  
**How it works:** Learns optimal policy regardless of behavior policy using maximum future value  
**Advantage:** Can learn optimal policy from any experience  
**Disadvantage:** Overestimation bias

#### **SARSA (On-Policy TD Learning)**  
**Update:** `Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]`  
**How it works:** Learns value of actual policy being followed  
**Advantage:** Conservative, learns true policy value  
**Disadvantage:** Limited to current policy

#### **Expected SARSA (Hybrid Approach)**
**Update:** `Q(s,a) = Q(s,a) + α[r + γ Σ π(a'|s') Q(s',a') - Q(s,a)]`  
**How it works:** Uses expected value over all actions weighted by policy  
**Advantage:** Reduced variance vs SARSA  
**Disadvantage:** More computationally expensive

### Advanced Exploration Methods

#### **Double Q-Learning (Bias Reduction)**
**Mechanism:** Maintains two Q-tables, randomly selects which to update  
**Innovation:** Separates action selection from evaluation to reduce overestimation  
**Trade-off:** Doubles memory, reduces bias

#### **UCB Q-Learning (Confidence-Based)**
**Selection:** `argmax[Q(s,a) + c√(ln(t)/N(s,a))]`  
**Innovation:** Principled exploration based on confidence intervals  
**Trade-off:** Systematic exploration, requires action count tracking

#### **Boltzmann Q-Learning (Temperature-Based)**
**Selection:** `P(a|s) = exp(Q(s,a)/T) / Σ exp(Q(s,a')/T)`  
**Innovation:** Smooth probabilistic action selection with temperature decay  
**Trade-off:** Natural exploration-exploitation transition, sensitive to scaling

### Enhanced Learning Methods

#### **N-Step Q-Learning (Multi-Step Bootstrap)**
**Returns:** Uses n-step returns bridging TD and Monte Carlo  
**Innovation:** Better credit assignment for delayed rewards  
**Trade-off:** Faster learning, higher variance

#### **Monte Carlo Control (Episodic)**  
**Returns:** Uses complete episode returns without bootstrapping  
**Innovation:** Unbiased value estimates  
**Trade-off:** No bias but high variance, episodic only

#### **Dyna-Q (Model-Based Planning)**
**Mechanism:** Learns environment model + performs planning updates  
**Innovation:** Sample efficiency through simulated experience  
**Trade-off:** Model learning overhead, assumes deterministic transitions

### Experience Replay Methods

#### **Experience Replay Q-Learning**
**Mechanism:** Stores experiences in buffer, samples random batches  
**Innovation:** Reuses experiences, breaks temporal correlation  
**Trade-off:** Memory overhead for sample efficiency

#### **Prioritized Experience Replay**
**Mechanism:** Prioritizes experiences by TD-error magnitude  
**Innovation:** Focuses on important transitions for faster learning  
**Trade-off:** Complex implementation, additional hyperparameters

### Algorithm Comparison Summary

| Algorithm | Type | Key Innovation | Best For |
|-----------|------|----------------|----------|
| **Q-Learning** | Off-Policy | Standard baseline | Education, comparison |
| **SARSA** | On-Policy | Conservative learning | Reliable performance |
| **Expected SARSA** | Hybrid | Expected values | Reduced variance |
| **Double Q-Learning** | Off-Policy | Bias reduction | Research, overestimation issues |
| **UCB Q-Learning** | Off-Policy | Principled exploration | Systematic exploration |
| **Boltzmann Q-Learning** | Off-Policy | Temperature exploration | Smooth exploration decay |
| **N-Step Q-Learning** | Off-Policy | Multi-step returns | Delayed reward environments |
| **Monte Carlo** | On-Policy | Unbiased estimates | Educational, episodic tasks |
| **Dyna-Q** | Off-Policy | Model-based planning | Sample efficiency research |
| **Experience Replay** | Off-Policy | Experience reuse | Sample efficiency |
| **Prioritized Replay** | Off-Policy | Intelligent sampling | Maximum performance |
