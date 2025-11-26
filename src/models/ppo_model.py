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

from environments.ldi_env import (
    LDIEnvironment,
    LDIContinuousEnvironment,
    LDIWideDiscreteEnvironment,
)
from config import plan_asset, planliab, n_sim, gamma


class BasePPOModel:
    """
    Shared functionality for PPO-based models operating on the LDI environments.
    """
    
    def __init__(
        self,
        *,
        environment_cls,
        model_path,
        model_label,
        tensorboard_log,
        n_envs=4,
        learning_rate=3e-4,
        n_steps=20,
        batch_size=32,
        n_epochs=20,
        verbose=1,
        max_steps=40,
        ent_coef=0.0
    ):
        """
        Initialize PPO model.
        
        Args:
            n_envs: Number of parallel environments for training
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            verbose: Verbosity level (0=none, 1=info, 2=debug)
            environment_cls: Environment class to instantiate for training and evaluation
            model_path: Relative path (under output/models) for saving the model
            model_label: Human-readable label for logging
            tensorboard_log: Directory for tensorboard logs
            max_steps: Number of steps per episode
        """
        self.n_envs = n_envs
        self.verbose = verbose
        self.max_steps = max_steps
        self.environment_cls = environment_cls
        self.model_label = model_label
        self.model_path = Path('output/models') / model_path

        # Create vectorized environment for training
        env = make_vec_env(
            lambda: environment_cls(sim_id=1, max_steps=self.max_steps),
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
            ent_coef=ent_coef,
            verbose=verbose,
            device="cpu",  # Use CPU for better performance with MLP
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])) 
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    def train(self, num_episodes, num_sims):
        """
        Train the PPO model.
        
        Args:
            num_episodes: Number of episodes to train
            num_sims: Number of simulations (used for compatibility)
        """
        # Convert episodes to timesteps (each episode has 40 steps)
        total_timesteps = num_episodes * self.max_steps
        
        print(f"Training {self.model_label} for {total_timesteps} timesteps ({num_episodes} episodes)...")
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        
        # Save the trained model
        self.model.save(str(self.model_path))
        print(f"Model saved as '{self.model_path}.zip'")
    
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
            eval_env = self.environment_cls(sim_id=i_sim, max_steps=self.max_steps)
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


class PPOModel(BasePPOModel):
    """
    Proximal Policy Optimization (PPO) model using the discrete-action LDI environment.
    """
    
    def __init__(
        self,
        n_envs=4,
        learning_rate=3e-4,
        #n_steps=2048,
        #batch_size=64,
        #n_epochs=10,
        verbose=1,
        max_steps=40,
        ent_coef=0.05
    ):
        super().__init__(
            environment_cls=LDIEnvironment,
            model_path="ppo_ldi_model",
            model_label="PPO",
            tensorboard_log="./logs/ppo_tensorboard/",
            n_envs=n_envs,
            learning_rate=learning_rate,
            #n_steps=n_steps,
            #batch_size=batch_size,
            #n_epochs=n_epochs,
            verbose=verbose,
            max_steps=max_steps,
            ent_coef=ent_coef
        )


class PPOContinuousModel(BasePPOModel):
    """
    PPO model variant that operates on the continuous-action LDI environment.
    """
    
    def __init__(
        self,
        n_envs=4,
        learning_rate=3e-4,
        #n_steps=2048,
        #batch_size=64,
        #n_epochs=10,
        verbose=1,
        max_steps=40,
        ent_coef=0.05
    ):
        super().__init__(
            environment_cls=LDIContinuousEnvironment,
            model_path="ppo_continuous_ldi_model",
            model_label="PPO (continuous)",
            tensorboard_log="./logs/ppo_continuous_tensorboard/",
            n_envs=n_envs,
            learning_rate=learning_rate,
            #n_steps=n_steps,
            #batch_size=batch_size,
            #n_epochs=n_epochs,
            verbose=verbose,
            max_steps=max_steps,
            ent_coef=ent_coef
        )


class PPOWideDiscreteModel(BasePPOModel):
    """
    PPO variant utilizing the wide-range discrete action environment.
    """
    
    def __init__(
        self,
        n_envs=4,
        learning_rate=3e-4,
        verbose=1,
        max_steps=40,
        ent_coef=0.05
    ):
        super().__init__(
            environment_cls=LDIWideDiscreteEnvironment,
            model_path="ppo_wide_discrete_ldi_model",
            model_label="PPO (wide discrete)",
            tensorboard_log="./logs/ppo_wide_discrete_tensorboard/",
            n_envs=n_envs,
            learning_rate=learning_rate,
            verbose=verbose,
            max_steps=max_steps,
            ent_coef=ent_coef
        )

