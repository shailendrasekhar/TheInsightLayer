"""
Multi-Algorithm Trainer for Frozen Lake

This script contains multiple reinforcement learning algorithms in one place
for easy comparison and experimentation. Includes Q-Learning, SARSA, and
other RL algorithms.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, List, Optional
from datetime import datetime


class BaseRLAgent:
    """
    Base class for reinforcement learning agents.
    Provides common functionality for training, evaluation, and model persistence.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize base RL agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions in the environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Training history
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],
            'steps_per_episode': [],
            'success_episodes': []
        }
        
        # Algorithm name (to be set by subclasses)
        self.algorithm_name = "BaseRL"
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If False, always use greedy policy
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def decay_epsilon(self):
        """Decay epsilon value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int, 
                      next_action: int = None, done: bool = False):
        """
        Update Q-value. To be implemented by subclasses.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (for SARSA)
            done: Whether episode is finished
        """
        raise NotImplementedError("Subclasses must implement update_q_value method")
    
    def train(self, env, episodes: int = 1000, verbose: bool = True, 
              print_every: int = 100) -> dict:
        """
        Train the agent using standard epsilon-greedy Q-learning approach.
        Can be overridden by subclasses for algorithm-specific logic.
        """
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],
            'steps_per_episode': [],
            'success_episodes': []
        }
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.update_q_value(state, action, reward, next_state, done=done)
                
                total_reward += reward
                steps += 1
                
                if reward > 0:
                    episode_success = True
                
                if done or truncated:
                    break
                
                state = next_state
            
            # Decay epsilon (if applicable)
            if hasattr(self, 'epsilon'):
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(getattr(self, 'epsilon', 0))
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                epsilon_val = getattr(self, 'epsilon', 0)
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={epsilon_val:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                epsilon_val = getattr(self, 'epsilon', 0)
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {epsilon_val:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history
    
    def evaluate(self, env, episodes: int = 100, verbose: bool = True) -> dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Gymnasium environment
            episodes: Number of evaluation episodes
            verbose: Whether to print evaluation progress
            
        Returns:
            Evaluation results dictionary
        """
        if verbose:
            print(f"\nEvaluating {self.algorithm_name} agent for {episodes} episodes...")
        
        total_rewards = []
        success_count = 0
        total_steps = 0
        
        for episode in range(episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.select_action(state, training=False)  # Greedy policy
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    if reward > 0:
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            total_steps += steps
        
        results = {
            'success_rate': success_count / episodes,
            'average_reward': np.mean(total_rewards),
            'average_steps': total_steps / episodes,
            'total_episodes': episodes
        }
        
        if verbose:
            print(f"{self.algorithm_name} Evaluation Results:")
            print(f"  Success Rate: {results['success_rate']:.1%}")
            print(f"  Average Reward: {results['average_reward']:.3f}")
            print(f"  Average Steps: {results['average_steps']:.1f}")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded agent instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create agent with loaded parameters
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon']
        )
        
        # Load Q-table and training history
        agent.q_table = model_data['q_table']
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent
    
    def get_policy(self) -> np.ndarray:
        """Get the current policy (best action for each state)."""
        return np.argmax(self.q_table, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """Get state values (max Q-value for each state)."""
        return np.max(self.q_table, axis=1)
    
    def reset_training_history(self):
        """Reset training history for a fresh start."""
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],
            'steps_per_episode': [],
            'success_episodes': []
        }
        self.epsilon = self.initial_epsilon
        print(f"{self.algorithm_name} training history reset.")
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history['episodes']:
            print("No training history available to plot.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = self.training_history['episodes']
        
        # 1. Rewards over time
        rewards = self.training_history['total_rewards']
        ax1.plot(episodes, rewards, alpha=0.3, label='Episode Rewards')
        
        # Moving average
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(episodes, moving_avg, label=f'Moving Average ({window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'{self.algorithm_name} Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Epsilon decay
        ax2.plot(episodes, self.training_history['epsilon_values'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title(f'{self.algorithm_name} Exploration Rate (Epsilon) Decay')
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate over time
        successes = self.training_history['success_episodes']
        window_size = min(50, len(successes) // 10)
        if window_size > 1:
            success_rate = pd.Series(successes).rolling(window=window_size).mean()
            ax3.plot(episodes, success_rate * 100)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title(f'{self.algorithm_name} Success Rate (Rolling Average, window={window_size})')
        ax3.grid(True, alpha=0.3)
        
        # 4. Steps per episode
        steps = self.training_history['steps_per_episode']
        ax4.plot(episodes, steps, alpha=0.3, label='Steps per Episode')
        
        if window_size > 1:
            steps_avg = pd.Series(steps).rolling(window=window_size).mean()
            ax4.plot(episodes, steps_avg, label=f'Moving Average ({window_size})')
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.set_title(f'{self.algorithm_name} Steps per Episode')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.algorithm_name} training progress plot saved to: {save_path}")
        
        plt.show()


class QLearningAgent(BaseRLAgent):
    """
    Q-Learning agent with detailed training progress tracking.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        super().__init__(n_states, n_actions, learning_rate, discount_factor, 
                        epsilon, epsilon_decay, min_epsilon)
        self.algorithm_name = "Q-Learning"
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int, 
                      next_action: int = None, done: bool = False):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Not used in Q-learning
            done: Whether episode is finished
        """
        if done:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def train(self, env, episodes: int = 1000, verbose: bool = True, 
              print_every: int = 100) -> dict:
        """
        Train the Q-learning agent.
        
        Args:
            env: Gymnasium environment
            episodes: Number of training episodes
            verbose: Whether to print training progress
            print_every: Print progress every N episodes
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Starting {self.algorithm_name} training for {episodes} episodes...")
            print(f"Initial epsilon: {self.epsilon:.3f}")
            print("-" * 60)
        
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done=terminated or truncated)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    if reward > 0:  # Success in reaching goal
                        episode_success = True
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={self.epsilon:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {self.epsilon:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history


class SarsaAgent(BaseRLAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent.
    On-policy algorithm that learns the policy being followed.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        super().__init__(n_states, n_actions, learning_rate, discount_factor, 
                        epsilon, epsilon_decay, min_epsilon)
        self.algorithm_name = "SARSA"
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int, 
                      next_action: int, done: bool = False):
        """
        Update Q-value using SARSA update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (required for SARSA)
            done: Whether episode is finished
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def train(self, env, episodes: int = 1000, verbose: bool = True, 
              print_every: int = 100) -> dict:
        """
        Train the SARSA agent.
        
        Args:
            env: Gymnasium environment
            episodes: Number of training episodes
            verbose: Whether to print training progress
            print_every: Print progress every N episodes
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Starting {self.algorithm_name} training for {episodes} episodes...")
            print(f"Initial epsilon: {self.epsilon:.3f}")
            print("-" * 60)
        
        for episode in range(episodes):
            state, info = env.reset()
            action = self.select_action(state, training=True)
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = self.select_action(next_state, training=True)
                
                # SARSA update (uses next_action instead of max)
                self.update_q_value(state, action, reward, next_state, next_action, 
                                  done=terminated or truncated)
                
                total_reward += reward
                steps += 1
                state = next_state
                action = next_action
                
                if terminated or truncated:
                    if reward > 0:
                        episode_success = True
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={self.epsilon:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {self.epsilon:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history


class ExpectedSarsaAgent(BaseRLAgent):
    """
    Expected SARSA agent.
    Uses expected value of next state-action pair instead of sample.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        super().__init__(n_states, n_actions, learning_rate, discount_factor, 
                        epsilon, epsilon_decay, min_epsilon)
        self.algorithm_name = "Expected-SARSA"
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int, 
                      next_action: int = None, done: bool = False):
        """
        Update Q-value using Expected SARSA update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Not used in Expected SARSA
            done: Whether episode is finished
        """
        if done:
            td_target = reward
        else:
            # Calculate expected value using current policy
            best_action = np.argmax(self.q_table[next_state])
            expected_value = 0.0
            
            for a in range(self.n_actions):
                if a == best_action:
                    prob = 1 - self.epsilon + self.epsilon / self.n_actions
                else:
                    prob = self.epsilon / self.n_actions
                expected_value += prob * self.q_table[next_state][a]
            
            td_target = reward + self.discount_factor * expected_value
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def train(self, env, episodes: int = 1000, verbose: bool = True, 
              print_every: int = 100) -> dict:
        """
        Train the Expected SARSA agent.
        """
        if verbose:
            print(f"Starting {self.algorithm_name} training for {episodes} episodes...")
            print(f"Initial epsilon: {self.epsilon:.3f}")
            print("-" * 60)
        
        for episode in range(episodes):
            state, info = env.reset()
            action = self.select_action(state, training=True)
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = self.select_action(next_state, training=True)
                
                # Expected SARSA update
                self.update_q_value(state, action, reward, next_state, done=terminated or truncated)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    if reward > 0:
                        episode_success = True
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={self.epsilon:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {self.epsilon:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history


class DoubleQLearningAgent(BaseRLAgent):
    """
    Double Q-Learning agent - reduces overestimation bias of standard Q-Learning.
    Uses two Q-tables and alternates updates to reduce maximization bias.
    """
    
    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Double-Q-Learning"
        
        # Initialize second Q-table for double Q-learning
        self.q_table2 = np.zeros((n_states, n_actions))
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy based on combined Q-tables."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            # Use average of both Q-tables for action selection
            combined_q = (self.q_table[state] + self.q_table2[state]) / 2
            return np.argmax(combined_q)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Update Q-values using Double Q-Learning algorithm."""
        if done:
            target = reward
        else:
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                # Update Q1, use Q2 for target calculation
                best_next_action = np.argmax(self.q_table[next_state])
                target = reward + self.discount_factor * self.q_table2[next_state, best_next_action]
                self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
            else:
                # Update Q2, use Q1 for target calculation  
                best_next_action = np.argmax(self.q_table2[next_state])
                target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
                self.q_table2[state, action] += self.learning_rate * (target - self.q_table2[state, action])
    
    def save_model(self, filepath: str):
        """Save model including both Q-tables."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'q_table2': self.q_table2,  # Save second Q-table
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including both Q-tables."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon']
        )
        
        agent.q_table = model_data['q_table']
        agent.q_table2 = model_data['q_table2']  # Load second Q-table
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class UCBQLearningAgent(BaseRLAgent):
    """
    Upper Confidence Bound (UCB) Q-Learning agent.
    Uses UCB for exploration instead of epsilon-greedy.
    """
    
    def __init__(self, n_states: int, n_actions: int, c: float = 2.0, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "UCB-Q-Learning"
        self.c = c  # UCB exploration parameter
        
        # Track action counts for UCB calculation
        self.action_counts = np.zeros((n_states, n_actions))
        self.total_steps = 0
    
    def choose_action(self, state: int) -> int:
        """Choose action using Upper Confidence Bound (UCB) exploration."""
        self.total_steps += 1
        
        # If any action hasn't been tried, try it
        if np.any(self.action_counts[state] == 0):
            untried_actions = np.where(self.action_counts[state] == 0)[0]
            return np.random.choice(untried_actions)
        
        # Calculate UCB values for all actions
        ucb_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            confidence = self.c * np.sqrt(np.log(self.total_steps) / self.action_counts[state, action])
            ucb_values[action] = self.q_table[state, action] + confidence
        
        return np.argmax(ucb_values)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Update Q-values using standard Q-Learning."""
        # Update action count
        self.action_counts[state, action] += 1
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def save_model(self, filepath: str):
        """Save model including action counts."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'action_counts': self.action_counts,
            'total_steps': self.total_steps,
            'c': self.c,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including action counts."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            c=model_data.get('c', 2.0)
        )
        
        agent.q_table = model_data['q_table']
        agent.action_counts = model_data['action_counts']
        agent.total_steps = model_data['total_steps']
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class BoltzmannQLearningAgent(BaseRLAgent):
    """
    Boltzmann (Softmax) Q-Learning agent.
    Uses temperature-based probabilistic action selection.
    """
    
    def __init__(self, n_states: int, n_actions: int, temperature: float = 1.0, 
                 temp_decay: float = 0.995, min_temperature: float = 0.01, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Boltzmann-Q-Learning"
        self.temperature = temperature
        self.initial_temperature = temperature
        self.temp_decay = temp_decay
        self.min_temperature = min_temperature
    
    def choose_action(self, state: int) -> int:
        """Choose action using Boltzmann (softmax) exploration."""
        if self.temperature <= 0:
            # If temperature is very low, choose greedily
            return np.argmax(self.q_table[state])
        
        # Calculate softmax probabilities
        q_values = self.q_table[state]
        exp_values = np.exp(q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action according to probabilities
        return np.random.choice(self.n_actions, p=probabilities)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Update Q-values using standard Q-Learning."""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def train(self, env, episodes: int = 1000, verbose: bool = True):
        """Train with temperature decay instead of epsilon decay."""
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],  # Will store temperature values for consistency
            'steps_per_episode': [],
            'success_episodes': []
        }
        
        print_every = 100
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.update_q_value(state, action, reward, next_state, done=done)
                
                total_reward += reward
                steps += 1
                
                if reward > 0:
                    episode_success = True
                
                if done or truncated:
                    break
                
                state = next_state
            
            # Decay temperature
            self.temperature = max(self.min_temperature, self.temperature * self.temp_decay)
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.temperature)  # Store temperature
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Temperature={self.temperature:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Temperature: {self.temperature:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save model including temperature parameters."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'temperature': self.temperature,
            'initial_temperature': self.initial_temperature,
            'temp_decay': self.temp_decay,
            'min_temperature': self.min_temperature,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including temperature parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            temperature=model_data.get('temperature', 1.0),
            temp_decay=model_data.get('temp_decay', 0.995),
            min_temperature=model_data.get('min_temperature', 0.01)
        )
        
        agent.q_table = model_data['q_table']
        agent.initial_temperature = model_data.get('initial_temperature', 1.0)
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class NStepQLearningAgent(BaseRLAgent):
    """
    N-Step Q-Learning agent.
    Uses n-step returns for updates, bridging TD and Monte Carlo methods.
    """
    
    def __init__(self, n_states: int, n_actions: int, n_steps: int = 3, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = f"{n_steps}-Step-Q-Learning"
        self.n_steps = n_steps
        
        # Store trajectory for n-step updates
        self.trajectory = []
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Store experience in trajectory for n-step updates."""
        # Add current experience to trajectory
        self.trajectory.append((state, action, reward, next_state, done))
        
        # Perform n-step update if we have enough experiences
        if len(self.trajectory) >= self.n_steps or done:
            self._perform_nstep_update()
    
    def _perform_nstep_update(self):
        """Perform n-step Q-learning update."""
        if not self.trajectory:
            return
        
        # Get the starting experience for update
        start_state, start_action, _, _, _ = self.trajectory[0]
        
        # Calculate n-step return
        n_step_return = 0
        for i, (_, _, reward, _, done) in enumerate(self.trajectory):
            n_step_return += (self.discount_factor ** i) * reward
            if done:
                break
        
        # Add bootstrapped value if episode didn't end
        if not self.trajectory[-1][4]:  # If not done
            final_state = self.trajectory[-1][3]
            n_step_return += (self.discount_factor ** len(self.trajectory)) * np.max(self.q_table[final_state])
        
        # Update Q-value
        current_q = self.q_table[start_state, start_action]
        self.q_table[start_state, start_action] += self.learning_rate * (n_step_return - current_q)
        
        # Remove the first experience from trajectory
        self.trajectory.pop(0)
    
    def train(self, env, episodes: int = 1000, verbose: bool = True):
        """Train with n-step updates."""
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],
            'steps_per_episode': [],
            'success_episodes': []
        }
        
        print_every = 100
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            # Clear trajectory at start of episode
            self.trajectory = []
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.update_q_value(state, action, reward, next_state, done=done)
                
                total_reward += reward
                steps += 1
                
                if reward > 0:
                    episode_success = True
                
                if done or truncated:
                    # Process remaining trajectory at episode end
                    while self.trajectory:
                        self._perform_nstep_update()
                    break
                
                state = next_state
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={self.epsilon:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {self.epsilon:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save model including n-step parameters."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'n_steps': self.n_steps,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including n-step parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            n_steps=model_data.get('n_steps', 3)
        )
        
        agent.q_table = model_data['q_table']
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class MonteCarloAgent(BaseRLAgent):
    """
    Monte Carlo Control agent.
    Uses complete episode returns for updates (no bootstrapping).
    """
    
    def __init__(self, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Monte-Carlo-Control"
        
        # Track returns for each state-action pair
        self.returns = {}
        for state in range(n_states):
            for action in range(n_actions):
                self.returns[(state, action)] = []
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Monte Carlo doesn't update during episode - only at the end."""
        pass  # Updates handled in train() method
    
    def train(self, env, episodes: int = 1000, verbose: bool = True):
        """Train using Monte Carlo method with complete episode returns."""
        self.training_history = {
            'episodes': [],
            'total_rewards': [],
            'epsilon_values': [],
            'steps_per_episode': [],
            'success_episodes': []
        }
        
        print_every = 100
        
        for episode in range(episodes):
            # Generate episode
            episode_data = []
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_success = False
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                episode_data.append((state, action, reward))
                total_reward += reward
                steps += 1
                
                if reward > 0:
                    episode_success = True
                
                if done or truncated:
                    break
                
                state = next_state
            
            # Calculate returns and update Q-values
            G = 0  # Return
            for t in reversed(range(len(episode_data))):
                state, action, reward = episode_data[t]
                G = self.discount_factor * G + reward
                
                # Check if this is the first occurrence (first-visit MC)
                if (state, action) not in [(s, a) for s, a, _ in episode_data[:t]]:
                    self.returns[(state, action)].append(G)
                    # Update Q-value as average of returns
                    self.q_table[state, action] = np.mean(self.returns[(state, action)])
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Store training history
            self.training_history['episodes'].append(episode + 1)
            self.training_history['total_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['steps_per_episode'].append(steps)
            self.training_history['success_episodes'].append(episode_success)
            
            # Print episode details
            if verbose:
                status = "SUCCESS" if episode_success else "FAILED"
                print(f"Episode {episode + 1:4d}: Reward={total_reward:.1f}, "
                      f"Steps={steps:2d}, Epsilon={self.epsilon:.3f}, Status={status}")
            
            # Print summary every N episodes
            if verbose and (episode + 1) % print_every == 0:
                recent_rewards = self.training_history['total_rewards'][-print_every:]
                recent_successes = self.training_history['success_episodes'][-print_every:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nSummary (Episodes {episode + 1 - print_every + 1}-{episode + 1}):")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Current Epsilon: {self.epsilon:.3f}")
                print("-" * 60)
        
        if verbose:
            print(f"{self.algorithm_name} training completed!")
            final_success_rate = np.mean(self.training_history['success_episodes'][-100:]) * 100
            print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
        
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save model including returns data."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'returns': self.returns,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including returns data."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon']
        )
        
        agent.q_table = model_data['q_table']
        agent.returns = model_data['returns']
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class DynaQAgent(BaseRLAgent):
    """
    Dyna-Q agent.
    Combines model-free learning with planning using a learned environment model.
    """
    
    def __init__(self, n_states: int, n_actions: int, planning_steps: int = 10, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Dyna-Q"
        self.planning_steps = planning_steps
        
        # Environment model: model[state][action] = (next_state, reward)
        self.model = {}
        
        # Track which state-action pairs have been visited
        self.visited_pairs = set()
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Update Q-values and environment model."""
        # Update Q-value (direct learning)
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        
        # Update model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (next_state, reward, done)
        self.visited_pairs.add((state, action))
        
        # Planning step
        self.plan()
    
    def plan(self):
        """Perform planning using the learned model."""
        for _ in range(self.planning_steps):
            if not self.visited_pairs:
                break
            
            # Random sample from visited state-action pairs
            visited_list = list(self.visited_pairs)
            idx = np.random.choice(len(visited_list))
            state, action = visited_list[idx]
            
            if state in self.model and action in self.model[state]:
                next_state, reward, done = self.model[state][action]
                
                # Update Q-value using model
                if done:
                    target = reward
                else:
                    target = reward + self.discount_factor * np.max(self.q_table[next_state])
                
                self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def save_model(self, filepath: str):
        """Save model including environment model and planning steps."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'model': self.model,
            'visited_pairs': self.visited_pairs,
            'planning_steps': self.planning_steps,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including environment model and planning steps."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            planning_steps=model_data.get('planning_steps', 10)
        )
        
        agent.q_table = model_data['q_table']
        agent.model = model_data['model']
        agent.visited_pairs = model_data['visited_pairs']
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class ExperienceReplayAgent(BaseRLAgent):
    """
    Q-Learning with Experience Replay.
    Stores experiences in a replay buffer and samples them for training.
    """
    
    def __init__(self, n_states: int, n_actions: int, buffer_size: int = 10000, 
                 batch_size: int = 32, replay_frequency: int = 4, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Experience-Replay-Q-Learning"
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_frequency = replay_frequency
        
        # Experience replay buffer
        self.replay_buffer = []
        self.step_count = 0
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def store_experience(self, state: int, action: int, reward: float, 
                        next_state: int, done: bool):
        """Store experience in replay buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)  # Remove oldest experience
        
        self.replay_buffer.append(experience)
    
    def replay_experiences(self):
        """Sample and replay experiences from buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state])
            
            self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Store experience and potentially replay."""
        # Store current experience
        self.store_experience(state, action, reward, next_state, done)
        
        # Immediate Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        
        # Replay experiences periodically
        self.step_count += 1
        if self.step_count % self.replay_frequency == 0:
            self.replay_experiences()
    
    def save_model(self, filepath: str):
        """Save model including replay buffer parameters."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'replay_buffer': self.replay_buffer,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'replay_frequency': self.replay_frequency,
            'step_count': self.step_count,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including replay buffer parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            buffer_size=model_data.get('buffer_size', 10000),
            batch_size=model_data.get('batch_size', 32),
            replay_frequency=model_data.get('replay_frequency', 4)
        )
        
        agent.q_table = model_data['q_table']
        agent.replay_buffer = model_data.get('replay_buffer', [])
        agent.step_count = model_data.get('step_count', 0)
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


class PrioritizedReplayAgent(BaseRLAgent):
    """
    Q-Learning with Prioritized Experience Replay.
    Prioritizes important experiences for more efficient learning.
    """
    
    def __init__(self, n_states: int, n_actions: int, buffer_size: int = 10000, 
                 batch_size: int = 32, replay_frequency: int = 4, 
                 alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.algorithm_name = "Prioritized-Replay-Q-Learning"
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_frequency = replay_frequency
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        
        # Experience replay buffer with priorities
        self.replay_buffer = []
        self.priorities = []
        self.step_count = 0
        self.max_priority = 1.0
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def store_experience(self, state: int, action: int, reward: float, 
                        next_state: int, done: bool, td_error: float = None):
        """Store experience with priority in replay buffer."""
        experience = (state, action, reward, next_state, done)
        
        # Calculate priority based on TD error
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)
        
        self.replay_buffer.append(experience)
        self.priorities.append(priority)
    
    def sample_batch(self):
        """Sample batch with prioritized sampling."""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.replay_buffer), 
                                 size=self.batch_size, 
                                 p=probabilities, 
                                 replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.replay_buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        
        # Get experiences
        batch = [self.replay_buffer[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def replay_experiences(self):
        """Sample and replay prioritized experiences."""
        batch, indices, weights = self.sample_batch()
        
        if batch is None:
            return
        
        td_errors = []
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state])
            
            current_q = self.q_table[state, action]
            td_error = target - current_q
            td_errors.append(td_error)
            
            # Apply importance sampling weight
            weighted_update = weights[i] * self.learning_rate * td_error
            self.q_table[state, action] += weighted_update
        
        # Update priorities
        self.update_priorities(indices, td_errors)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int = None, done: bool = False):
        """Store experience with TD error and potentially replay."""
        # Calculate TD error for priority
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        current_q = self.q_table[state, action]
        td_error = target - current_q
        
        # Store experience with priority
        self.store_experience(state, action, reward, next_state, done, td_error)
        
        # Immediate Q-learning update
        self.q_table[state, action] += self.learning_rate * td_error
        
        # Replay experiences periodically
        self.step_count += 1
        if self.step_count % self.replay_frequency == 0:
            self.replay_experiences()
    
    def save_model(self, filepath: str):
        """Save model including prioritized replay parameters."""
        model_data = {
            'algorithm_name': self.algorithm_name,
            'q_table': self.q_table,
            'replay_buffer': self.replay_buffer,
            'priorities': self.priorities,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'replay_frequency': self.replay_frequency,
            'alpha': self.alpha,
            'beta': self.beta,
            'step_count': self.step_count,
            'max_priority': self.max_priority,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.algorithm_name} model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model including prioritized replay parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_decay=model_data['epsilon_decay'],
            min_epsilon=model_data['min_epsilon'],
            buffer_size=model_data.get('buffer_size', 10000),
            batch_size=model_data.get('batch_size', 32),
            replay_frequency=model_data.get('replay_frequency', 4),
            alpha=model_data.get('alpha', 0.6),
            beta=model_data.get('beta', 0.4)
        )
        
        agent.q_table = model_data['q_table']
        agent.replay_buffer = model_data.get('replay_buffer', [])
        agent.priorities = model_data.get('priorities', [])
        agent.step_count = model_data.get('step_count', 0)
        agent.max_priority = model_data.get('max_priority', 1.0)
        agent.initial_epsilon = model_data['initial_epsilon']
        agent.training_history = model_data['training_history']
        
        print(f"Model loaded from: {filepath}")
        return agent


def compare_algorithms():
    """Compare all available RL algorithms"""
    print("Comparing Multiple RL Algorithms")
    print("=" * 60)
    
    # Environment setup
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode=None)
    
    algorithms = {
        'Q-Learning': QLearningAgent,
        'SARSA': SarsaAgent,
        'Expected-SARSA': ExpectedSarsaAgent,
        'Double-Q-Learning': DoubleQLearningAgent,
        'UCB-Q-Learning': UCBQLearningAgent,
        'Boltzmann-Q-Learning': BoltzmannQLearningAgent,
        '3-Step-Q-Learning': lambda **kwargs: NStepQLearningAgent(n_steps=3, **kwargs),
        'Monte-Carlo-Control': MonteCarloAgent,
        'Dyna-Q': DynaQAgent,
        'Experience-Replay': ExperienceReplayAgent,
        'Prioritized-Replay': PrioritizedReplayAgent
    };
    
    results = [];
    training_histories = {};
    models_dir = '/home/shailendrasekhar/Projects/TheInsightLayer/games/frozen_lake/models';
    os.makedirs(models_dir, exist_ok=True);
    
    for alg_name, AlgorithmClass in algorithms.items():
        print(f"\nTraining {alg_name}...")
        
        agent = AlgorithmClass(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        
        # Train
        training_history = agent.train(env, episodes=1000, verbose=False);
        training_histories[alg_name] = training_history;
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S");
        model_filename = f"{alg_name.lower().replace('-', '_')}_model_{timestamp}.pkl";
        model_path = os.path.join(models_dir, model_filename);
        agent.save_model(model_path);
        
        # Evaluate
        eval_results = agent.evaluate(env, episodes=100, verbose=False);
        
        results.append({
            'Algorithm': alg_name,
            'Success Rate': f"{eval_results['success_rate']:.1%}",
            'Average Reward': f"{eval_results['average_reward']:.3f}",
            'Final Epsilon': f"{agent.epsilon:.3f}",
            'Model File': model_filename
        })
    
    env.close();
    
    # Print results
    print(f"\n{'='*60}");
    print("ALGORITHM COMPARISON RESULTS");
    print(f"{'='*60}");
    df = pd.DataFrame(results);
    print(df.to_string(index=False));
    
    # Plot comparison
    plot_algorithm_comparison(training_histories);
    
    return results, training_histories;


def plot_algorithm_comparison(training_histories):
    """Plot comparison of training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (alg_name, history) in enumerate(training_histories.items()):
        color = colors[i % len(colors)]
        episodes = history['episodes']
        
        # Rewards
        rewards = history['total_rewards']
        window_size = 50
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(episodes, rewards_smooth, label=f'{alg_name}', color=color)
        
        # Epsilon
        ax2.plot(episodes, history['epsilon_values'], label=f'{alg_name}', color=color)
        
        # Success rate
        successes = history['success_episodes']
        if len(successes) >= window_size:
            success_rate = pd.Series(successes).rolling(window=window_size).mean()
            ax3.plot(episodes, success_rate * 100, label=f'{alg_name}', color=color)
        
        # Steps per episode
        steps = history['steps_per_episode']
        if len(steps) >= window_size:
            steps_smooth = pd.Series(steps).rolling(window=window_size).mean()
            ax4.plot(episodes, steps_smooth, label=f'{alg_name}', color=color)
    
    # Configure plots
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Rewards Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Steps')
    ax4.set_title('Steps per Episode Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    models_dir = '/home/shailendrasekhar/Projects/TheInsightLayer/games/frozen_lake/models'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(models_dir, f'algorithm_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Algorithm comparison plot saved to: {save_path}")
    
    plt.show()


def demo_multiple_hyperparameters():
    """Demonstrate how hyperparameters affect learning"""
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 50)
    
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode=None)
    
    # Different learning rates to test
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    results = []
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        
        # Train with fewer episodes for quick demo
        agent.train(env, episodes=500, verbose=False)
        eval_results = agent.evaluate(env, episodes=100, verbose=False)
        
        results.append({
            'Learning Rate': lr,
            'Success Rate': f"{eval_results['success_rate']:.1%}",
            'Average Reward': f"{eval_results['average_reward']:.3f}"
        })
    
    env.close()
    
    print(f"\n{'='*50}")
    print("HYPERPARAMETER RESULTS")
    print(f"{'='*50}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def main():
    """Main function for multi-algorithm demo"""
    print("Multi-Algorithm Frozen Lake Demo")
    print("Choose a demo:")
    print("1. Compare Q-Learning vs SARSA vs Expected-SARSA")
    print("2. Hyperparameter sensitivity analysis")
    print("3. Train single algorithm (with detailed progress)")
    print("4. Load and test saved model")
    print("5. Run all demos")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == '1':
        compare_algorithms()
    elif choice == '2':
        demo_multiple_hyperparameters()
    elif choice == '3':
        train_single_algorithm()
    elif choice == '4':
        load_and_test_model()
    elif choice == '5':
        compare_algorithms()
        print("\n" + "="*60 + "\n")
        demo_multiple_hyperparameters()
    else:
        print("Invalid choice. Running algorithm comparison.")
        compare_algorithms()


def train_single_algorithm():
    """Train a single algorithm with detailed progress"""
    print("Available algorithms:")
    algorithms = {
        '1': ('Q-Learning', QLearningAgent),
        '2': ('SARSA', SarsaAgent), 
        '3': ('Expected-SARSA', ExpectedSarsaAgent)
    }
    
    for key, (name, _) in algorithms.items():
        print(f"{key}. {name}")
    
    choice = input("Choose algorithm (1-3): ").strip()
    
    if choice not in algorithms:
        print("Invalid choice. Using Q-Learning.")
        choice = '1'
    
    alg_name, AlgorithmClass = algorithms[choice]
    
    # Environment setup
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode=None)
    
    agent = AlgorithmClass(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    print(f"\nTraining {alg_name} with detailed progress...")
    episodes = int(input("Number of episodes (default 1000): ") or "1000")
    
    # Train with detailed progress
    agent.train(env, episodes=episodes, verbose=True, print_every=50)
    
    # Evaluate
    agent.evaluate(env, episodes=100, verbose=True)
    
    # Save model
    models_dir = '/home/shailendrasekhar/Projects/TheInsightLayer/games/frozen_lake/models'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{alg_name.lower().replace('-', '_')}_model_{timestamp}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    agent.save_model(model_path)
    
    env.close()


def load_and_test_model():
    """Load and test a saved model"""
    models_dir = '/home/shailendrasekhar/Projects/TheInsightLayer/games/frozen_lake/models'
    
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not model_files:
        print("No saved models found.")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    try:
        choice = int(input("Enter model number: ")) - 1
        model_path = os.path.join(models_dir, model_files[choice])
    except (ValueError, IndexError):
        print("Invalid choice.")
        return
    
    # Try to determine algorithm type from filename
    if 'q_learning' in model_files[choice] or 'qlearning' in model_files[choice]:
        agent = QLearningAgent.load_model(model_path)
    elif 'sarsa' in model_files[choice] and 'expected' not in model_files[choice]:
        agent = SarsaAgent.load_model(model_path)
    elif 'expected' in model_files[choice]:
        agent = ExpectedSarsaAgent.load_model(model_path)
    else:
        # Default to Q-Learning
        agent = QLearningAgent.load_model(model_path)
    
    # Test the loaded model
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode=None)
    
    print(f"\nTesting loaded {agent.algorithm_name} model...")
    agent.evaluate(env, episodes=100, verbose=True)
    
    # Show some sample episodes
    print(f"\nSample episodes using {agent.algorithm_name}:")
    for episode in range(3):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            print(f"  Step {steps}: State {state} -> Action {action} -> State {next_state}, Reward {reward}")
            
            if terminated or truncated:
                status = "SUCCESS" if reward > 0 else "FAILED"
                print(f"  Episode ended: {status}, Total reward: {total_reward}, Steps: {steps}")
                break
                
            state = next_state
    
    env.close()
