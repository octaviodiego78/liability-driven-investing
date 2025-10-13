"""
PPO model implementation using stable-baselines3.
Uses the shared LDI environment.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.ldi_env import LDIEnvironment
from config import plan_asset, planliab, n_sim, gamma


class PPOModel:
    """
    Proximal Policy Optimization (PPO) model for LDI.
    Uses stable-baselines3 implementation with the shared LDI environment.
    """
    
    def __init__(self, n_envs=4, learning_rate=3e-4, n_steps=2048, 
                 batch_size=64, n_epochs=10, verbose=1):
        """
        Initialize PPO model.
        
        Args:
            n_envs: Number of parallel environments for training
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        self.n_envs = n_envs
        self.verbose = verbose
        
        # Create vectorized environment for training
        env = make_vec_env(
            lambda: LDIEnvironment(sim_id=1, max_steps=40),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )
        
        self.env = VecMonitor(env)
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=verbose,
            device="cpu",  # Use CPU for better performance with MLP
            tensorboard_log="./logs/ppo_tensorboard/"
        )
    
    def train(self, num_episodes, num_sims):
        """
        Train the PPO model.
        
        Args:
            num_episodes: Number of episodes to train
            num_sims: Number of simulations (used for compatibility)
        """
        # Convert episodes to timesteps (each episode has 40 steps)
        total_timesteps = num_episodes * 40
        
        print(f"Training PPO for {total_timesteps} timesteps ({num_episodes} episodes)...")
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path('output/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the trained model
        self.model.save("output/models/ppo_ldi_model")
        print("Model saved as 'output/models/ppo_ldi_model.zip'")
    
    def evaluate(self, num_sims):
        """
        Evaluate the trained model across multiple simulations.
        
        Args:
            num_sims: Number of simulations to run
            
        Returns:
            tuple: (tliabArray, tsaa, sims, qs, rwds, rwds_total, sims_sim)
                Same format as existing DQN models for compatibility
        """
        # Initialize tracking arrays
        tsaa = [0.5, 0.5]
        tliabArray = np.array([1, 0, planliab, 0, 0, 0])
        tliabArray = np.append(tliabArray, np.multiply([0.5, 0.5], plan_asset))
        tliabArray = np.append(tliabArray, np.array([
            plan_asset, 
            plan_asset/planliab, 
            plan_asset-planliab
        ]))
        
        sims = []
        qs = []
        rwds = []
        rwds_total = []
        sims_sim = []
        
        # Run evaluation for each simulation
        for i_sim in range(1, num_sims + 1):
            # Create environment for this simulation
            eval_env = LDIEnvironment(sim_id=i_sim, max_steps=40)
            obs, _ = eval_env.reset()
            
            reward_total = 0
            done = False
            step = 0
            
            while not done:
                # Predict action (deterministic for evaluation)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                reward_total += reward * (gamma ** step)
                
                # Store results
                tsaa = np.vstack([tsaa, np.array(info['saa'])])
                tliabArray = np.vstack([tliabArray, eval_env.liabArray])
                
                sims.append(i_sim)
                qs.append(step + 1)
                rwds.append(float(reward))
                
                step += 1
            
            rwds_total.append(reward_total)
            sims_sim.append(i_sim)
            
            if i_sim % 50 == 0:
                print(f"Simulation: {i_sim}/{num_sims}")
        
        return tliabArray, tsaa, sims, qs, rwds, rwds_total, sims_sim

