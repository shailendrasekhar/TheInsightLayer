"""
Frozen Lake Environment Demo using Gymnasium

This file demonstrates various aspects of the Frozen Lake environment:
1. Basic environment setup and random play
2. Environment visualization 
3. Q-learning training with detailed progress tracking
4. Model saving and loading functionality
5. Policy visualization and analysis
6. Performance evaluation and comparison

The Frozen Lake environment is a         elif choice == '2':
            self.agent = self.train_agent(episodes=1000, algorithm='SARSA')
        elif choice == '3':
            self.agent = self.train_agent(episodes=1000, algorithm='Expected-SARSA')sic reinforcement learning problem
where an agent must navigate from start to goal on a frozen lake while
avoiding holes in the ice.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import os
from datetime import datetime
from multi_algorithm_demo import (
    QLearningAgent, SarsaAgent, ExpectedSarsaAgent,
    DoubleQLearningAgent, UCBQLearningAgent, BoltzmannQLearningAgent,
    NStepQLearningAgent, MonteCarloAgent, DynaQAgent,
    ExperienceReplayAgent, PrioritizedReplayAgent
)

class FrozenLakeDemo:
    def __init__(self, map_name='4x4', is_slippery=True, render_mode=None):
        """
        Initialize the Frozen Lake environment
        
        Args:
            map_name: '4x4' or '8x8' 
            is_slippery: If True, actions have stochastic outcomes
            render_mode: 'human', 'ansi', 'rgb_array', or None (no rendering)
        """
        self.env = gym.make('FrozenLake-v1', 
                           map_name=map_name, 
                           is_slippery=is_slippery,
                           render_mode=render_mode)
        self.map_name = map_name
        self.is_slippery = is_slippery
        
        # Environment info
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # Create models directory
        self.models_dir = '/home/shailendrasekhar/Projects/TheInsightLayer/games/frozen_lake/models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize agent (will be created when needed)
        self.agent = None
        
        print(f"Environment: FrozenLake-{map_name}")
        print(f"Slippery: {is_slippery}")
        print(f"Render mode: {render_mode if render_mode else 'None (faster training)'}")
        print(f"States: {self.n_states}")
        print(f"Actions: {self.n_actions}")
        print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")
        print(f"Models will be saved to: {self.models_dir}")
        
    def demo_random_play(self, episodes=5):
        """Demonstrate random gameplay"""
        print("\n=== Random Play Demo ===")
        
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}:")
            print(f"Starting state: {state}")
            
            while True:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                
                print(f"  Step {steps}: Action={action}, State={state}->{next_state}, Reward={reward}")
                
                if terminated or truncated:
                    print(f"  Episode ended: Total reward={total_reward}, Steps={steps}")
                    if terminated and reward > 0:
                        print("  SUCCESS! Reached the goal!")
                    elif terminated:
                        print("  FAILED! Fell into a hole!")
                    break
                    
                state = next_state
                
    def visualize_environment(self):
        """Visualize the environment layout"""
        print("\n=== Environment Layout ===")
        
        # Get the map description
        if hasattr(self.env.unwrapped, 'desc'):
            desc = self.env.unwrapped.desc
            print("Map layout:")
            for row in desc:
                print(''.join([char.decode('utf-8') for char in row]))
        
        print("\nLegend:")
        print("S = Start position")
        print("F = Frozen surface (safe)")
        print("H = Hole (game over)")
        print("G = Goal (win)")
        
    def train_agent(self, episodes=1000, learning_rate=0.1, discount_factor=0.95,
                   epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, 
                   save_model=True, verbose=True, algorithm='Q-Learning'):
        """
        Train a reinforcement learning agent using the specified algorithm
        
        Args:
            episodes: Number of training episodes
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
            save_model: Whether to save the trained model
            verbose: Whether to print detailed training progress
            algorithm: Algorithm to use ('Q-Learning', 'SARSA', 'Expected-SARSA')
            
        Returns:
            Trained agent instance
        """
        print(f"\n=== Training {algorithm} Agent ({episodes} episodes) ===")
        
        # Choose algorithm class
        algorithm_classes = {
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
        }
        
        if algorithm not in algorithm_classes:
            print(f"Unknown algorithm: {algorithm}. Using Q-Learning.")
            algorithm = 'Q-Learning'
        
        AgentClass = algorithm_classes[algorithm]
        
        # Create agent
        self.agent = AgentClass(
            n_states=self.n_states,
            n_actions=self.n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon
        )
        
        # Train the agent
        training_history = self.agent.train(
            env=self.env, 
            episodes=episodes, 
            verbose=verbose,
            print_every=100
        )
        
        # Save model if requested
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slippery_str = "slippery" if self.is_slippery else "deterministic"
            algorithm_str = algorithm.lower().replace('-', '_')
            model_filename = f"{algorithm_str}_{self.map_name}_{slippery_str}_{timestamp}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            self.agent.save_model(model_path)
        
        return self.agent
    
    def load_agent(self, model_path=None):
        """
        Load a pre-trained agent
        
        Args:
            model_path: Path to the saved model. If None, lists available models.
            
        Returns:
            Loaded agent instance
        """
        if model_path is None:
            # List available models
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            if not model_files:
                print("No saved models found.")
                return None
            
            print("Available models:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            
            choice = input("Enter model number to load (or 'none' to skip): ").strip()
            if choice.lower() == 'none':
                return None
            
            try:
                model_idx = int(choice) - 1
                model_path = os.path.join(self.models_dir, model_files[model_idx])
                selected_file = model_files[model_idx]
            except (ValueError, IndexError):
                print("Invalid choice.")
                return None
        else:
            selected_file = os.path.basename(model_path)
        
        try:
            # Determine algorithm type from filename
            if 'double_q' in selected_file or 'double-q' in selected_file:
                self.agent = DoubleQLearningAgent.load_model(model_path)
            elif 'ucb' in selected_file:
                self.agent = UCBQLearningAgent.load_model(model_path)
            elif 'boltzmann' in selected_file:
                self.agent = BoltzmannQLearningAgent.load_model(model_path)
            elif 'step' in selected_file and 'q' in selected_file:
                self.agent = NStepQLearningAgent.load_model(model_path)
            elif 'monte' in selected_file or 'mc' in selected_file:
                self.agent = MonteCarloAgent.load_model(model_path)
            elif 'dyna' in selected_file:
                self.agent = DynaQAgent.load_model(model_path)
            elif 'experience' in selected_file and 'prioritized' not in selected_file:
                self.agent = ExperienceReplayAgent.load_model(model_path)
            elif 'prioritized' in selected_file:
                self.agent = PrioritizedReplayAgent.load_model(model_path)
            elif 'q_learning' in selected_file or 'qlearning' in selected_file:
                self.agent = QLearningAgent.load_model(model_path)
            elif 'sarsa' in selected_file and 'expected' not in selected_file:
                self.agent = SarsaAgent.load_model(model_path)
            elif 'expected' in selected_file:
                self.agent = ExpectedSarsaAgent.load_model(model_path)
            else:
                # Default to Q-Learning
                self.agent = QLearningAgent.load_model(model_path)
            
            return self.agent
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def evaluate_policy(self, episodes=100):
        """Evaluate the trained agent's policy"""
        if self.agent is None:
            print("No trained agent available. Please train or load an agent first.")
            return None, None
        
        print(f"\n=== Policy Evaluation ({episodes} episodes) ===")
        
        results = self.agent.evaluate(self.env, episodes=episodes, verbose=True)
        return results['success_rate'], results['average_reward']
    
    def visualize_q_table(self):
        """Visualize the agent's Q-table as heatmaps"""
        if self.agent is None:
            print("No trained agent available. Please train or load an agent first.")
            return
        
        print(f"\n=== {self.agent.algorithm_name} Q-Table Visualization ===")
        
        q_table = self.agent.q_table
        
        if self.map_name == '4x4':
            grid_size = 4
        else:
            grid_size = 8
            
        # Create subplots for each action
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        actions = ['Left', 'Down', 'Right', 'Up']
        
        for i, (ax, action) in enumerate(zip(axes.flat, actions)):
            q_values = q_table[:, i].reshape(grid_size, grid_size)
            sns.heatmap(q_values, annot=True, fmt='.2f', ax=ax, 
                       cmap='viridis', center=0)
            ax.set_title(f'{self.agent.algorithm_name} Q-values for Action: {action}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_str = self.agent.algorithm_name.lower().replace('-', '_')
        filename = f'{algorithm_str}_q_table_heatmap_{self.map_name}_{timestamp}.png'
        filepath = os.path.join(self.models_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Q-table heatmap saved to: {filepath}")
        
        plt.show()
        
    def visualize_policy(self):
        """Visualize the learned policy"""
        if self.agent is None:
            print("No trained agent available. Please train or load an agent first.")
            return
        
        print(f"\n=== {self.agent.algorithm_name} Policy Visualization ===")
        
        q_table = self.agent.q_table
        
        if self.map_name == '4x4':
            grid_size = 4
        else:
            grid_size = 8
        
        # Get optimal actions and state values
        policy = self.agent.get_policy()
        state_values = self.agent.get_state_values()
        
        policy_grid = policy.reshape(grid_size, grid_size)
        value_grid = state_values.reshape(grid_size, grid_size)
        
        # Create arrow symbols for actions
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create heatmap of state values
        sns.heatmap(value_grid, annot=False, ax=ax, cmap='Blues', alpha=0.7)
        
        # Add policy arrows
        for i in range(grid_size):
            for j in range(grid_size):
                action = policy_grid[i, j]
                symbol = action_symbols[action]
                ax.text(j + 0.5, i + 0.5, symbol, ha='center', va='center', 
                       fontsize=20, fontweight='bold', color='red')
        
        ax.set_title(f'{self.agent.algorithm_name} Policy (Arrows) and State Values (Colors)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_str = self.agent.algorithm_name.lower().replace('-', '_')
        filename = f'{algorithm_str}_policy_visualization_{self.map_name}_{timestamp}.png'
        filepath = os.path.join(self.models_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to: {filepath}")
        
        plt.show()
        
    def plot_training_progress(self):
        """Plot training progress using agent's built-in method"""
        if self.agent is None:
            print("No trained agent available. Please train or load an agent first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_str = self.agent.algorithm_name.lower().replace('-', '_')
        filename = f'{algorithm_str}_training_progress_{self.map_name}_{timestamp}.png'
        save_path = os.path.join(self.models_dir, filename)
        
        self.agent.plot_training_progress(save_path=save_path)
        
    def demo_trained_agent(self, episodes=3, render=True):
        """Demonstrate the trained agent playing"""
        if self.agent is None:
            print("No trained agent available. Please train or load an agent first.")
            return
        
        print(f"\n=== Trained Agent Demo ({episodes} episodes) ===")
        
        # Create a separate environment for demonstration with rendering if requested
        if render and self.env.render_mode is None:
            demo_env = gym.make('FrozenLake-v1', 
                              map_name=self.map_name, 
                              is_slippery=self.is_slippery,
                              render_mode='human')
        else:
            demo_env = self.env
        
        for episode in range(episodes):
            state, info = demo_env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}:")
            print(f"Starting state: {state}")
            
            while True:
                action = self.agent.select_action(state, training=False)  # Use greedy policy
                next_state, reward, terminated, truncated, info = demo_env.step(action)
                
                total_reward += reward
                steps += 1
                
                print(f"  Step {steps}: Action={action}, State={state}->{next_state}, Reward={reward}")
                
                if terminated or truncated:
                    print(f"  Episode ended: Total reward={total_reward}, Steps={steps}")
                    if terminated and reward > 0:
                        print("  SUCCESS! Reached the goal!")
                    elif terminated:
                        print("  FAILED! Fell into a hole!")
                    break
                    
                state = next_state
                if render:
                    time.sleep(0.5)  # Pause for visualization only when rendering
        
        # Close demo environment if we created a separate one
        if render and demo_env != self.env:
            demo_env.close()
                
    def run_complete_demo(self):
        """Run a complete demonstration"""
        print("Starting Complete Frozen Lake Demo")
        print("=" * 50)
        
        # 1. Visualize environment
        self.visualize_environment()
        
        # 2. Random play demo
        self.demo_random_play(episodes=3)
        
        # 3. Check if user wants to load existing model or train new one
        print("\n" + "=" * 50)
        print("TRAINING OPTIONS")
        print("=" * 50)
        print("1. Train new Q-Learning agent")
        print("2. Train new SARSA agent")
        print("3. Train new Expected-SARSA agent")
        print("4. Train new Double-Q-Learning agent")
        print("5. Train new UCB-Q-Learning agent")
        print("6. Train new Boltzmann-Q-Learning agent")
        print("7. Train new 3-Step-Q-Learning agent")
        print("8. Train new Monte-Carlo-Control agent")
        print("9. Train new Dyna-Q agent")
        print("10. Train new Experience-Replay agent")
        print("11. Train new Prioritized-Replay agent")
        print("12. Load existing agent")
        print("13. Train new agent (fast - 500 episodes)")
        
        choice = input("Choose option (1-13): ").strip()
        
        if choice == '2':
            self.agent = self.train_agent(episodes=1000, algorithm='SARSA')
        elif choice == '3':
            self.agent = self.train_agent(episodes=1000, algorithm='Expected-SARSA')
        elif choice == '4':
            self.agent = self.train_agent(episodes=1000, algorithm='Double-Q-Learning')
        elif choice == '5':
            self.agent = self.train_agent(episodes=1000, algorithm='UCB-Q-Learning')
        elif choice == '6':
            self.agent = self.train_agent(episodes=1000, algorithm='Boltzmann-Q-Learning')
        elif choice == '7':
            self.agent = self.train_agent(episodes=1000, algorithm='3-Step-Q-Learning')
        elif choice == '8':
            self.agent = self.train_agent(episodes=1000, algorithm='Monte-Carlo-Control')
        elif choice == '9':
            self.agent = self.train_agent(episodes=1000, algorithm='Dyna-Q')
        elif choice == '10':
            self.agent = self.train_agent(episodes=1000, algorithm='Experience-Replay')
        elif choice == '11':
            self.agent = self.train_agent(episodes=1000, algorithm='Prioritized-Replay')
        elif choice == '12':
            self.agent = self.load_agent()
            if self.agent is None:
                print("No agent loaded. Training new Q-Learning agent...")
                self.agent = self.train_agent(episodes=1000)
        elif choice == '13':
            self.agent = self.train_agent(episodes=500)
        else:
            self.agent = self.train_agent(episodes=1000)
        
        # 4. Evaluate policy
        success_rate, avg_reward = self.evaluate_policy(episodes=100)
        
        # 5. Visualizations
        self.plot_training_progress()
        self.visualize_q_table()
        self.visualize_policy()
        
        # 6. Demo trained agent (ask if user wants to see rendering)
        show_demo = input("\nShow visual demo of trained agent? (y/n): ").strip().lower()
        if show_demo == 'y':
            self.demo_trained_agent(episodes=3, render=True)
        else:
            self.demo_trained_agent(episodes=3, render=False)
        
        # 7. Summary
        print("\n" + "=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        print(f"Environment: FrozenLake-{self.map_name}")
        print(f"Slippery: {self.is_slippery}")
        if success_rate is not None and avg_reward is not None:
            print(f"Final Success Rate: {success_rate:.1%}")
            print(f"Final Average Reward: {avg_reward:.3f}")
        
        if self.agent and self.agent.training_history['episodes']:
            total_episodes = len(self.agent.training_history['episodes'])
            print(f"Total Training Episodes: {total_episodes}")
            print(f"Algorithm Used: {self.agent.algorithm_name}")
        
        self.env.close()
        
        return self.agent

def main():
    """Main function to run different demos"""
    print("Frozen Lake Environment Demos")
    print("Choose a demo to run:")
    print("1. 4x4 Grid (Slippery)")
    print("2. 4x4 Grid (Not Slippery)")
    print("3. 8x8 Grid (Slippery)")
    print("4. Compare All Configurations")
    print("5. Load and test existing model")
    print("6. Multi-algorithm comparison (All 11 algorithms)")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == '1':
        demo = FrozenLakeDemo(map_name='4x4', is_slippery=True, render_mode=None)
        demo.run_complete_demo()
        
    elif choice == '2':
        demo = FrozenLakeDemo(map_name='4x4', is_slippery=False, render_mode=None)
        demo.run_complete_demo()
        
    elif choice == '3':
        demo = FrozenLakeDemo(map_name='8x8', is_slippery=True, render_mode=None)
        demo.run_complete_demo()
        
    elif choice == '4':
        # Compare all configurations
        configs = [
            ('4x4', True, '4x4 Slippery'),
            ('4x4', False, '4x4 Non-Slippery'),
            ('8x8', True, '8x8 Slippery')
        ]
        
        results = []
        
        for map_name, is_slippery, name in configs:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            
            demo = FrozenLakeDemo(map_name=map_name, is_slippery=is_slippery, render_mode=None)
            agent = demo.train_agent(episodes=1000, verbose=False)
            eval_results = agent.evaluate(demo.env, episodes=100, verbose=False)
            
            results.append({
                'Configuration': name,
                'Algorithm': agent.algorithm_name,
                'Success Rate': f"{eval_results['success_rate']:.1%}",
                'Average Reward': f"{eval_results['average_reward']:.3f}",
                'Training Episodes': len(agent.training_history['episodes'])
            })
            
            demo.env.close()
        
        # Print comparison table
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
    elif choice == '5':
        # Load and test existing model
        demo = FrozenLakeDemo(map_name='4x4', is_slippery=True, render_mode=None)
        agent = demo.load_agent()
        
        if agent is not None:
            print(f"\nTesting loaded {agent.algorithm_name} model...")
            demo.evaluate_policy(episodes=100)
            
            # Ask if user wants visual demo
            show_demo = input("Show visual demo of loaded agent? (y/n): ").strip().lower()
            demo.demo_trained_agent(episodes=3, render=(show_demo == 'y'))
            demo.visualize_policy()
        
        demo.env.close()
        
    elif choice == '6':
        # Multi-algorithm comparison
        from multi_algorithm_demo import compare_algorithms
        compare_algorithms()
        
    else:
        print("Invalid choice. Running default 4x4 slippery demo.")
        demo = FrozenLakeDemo(map_name='4x4', is_slippery=True, render_mode=None)
        demo.run_complete_demo()

if __name__ == "__main__":
    main()
