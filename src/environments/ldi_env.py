"""
Gym environment for Liability-Driven Investing (LDI) simulation.
Compatible with stable-baselines3 and other RL libraries.

This environment extracts the simulation logic that was originally embedded
in the DQN model classes, making it reusable across different RL algorithms.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import (gamma, n_dyn, plan_asset, planliab, employees, 
                    var, fundmap, tmpArrays, liabAll, multiple, neg_multiple)


class LDIEnvironment(gym.Env):
    """
    Custom Gym Environment for Liability-Driven Investing.
    
    This environment simulates a pension fund managing assets to meet liabilities.
    The agent adjusts asset allocation between AA-rated bonds and public equity.
    
    Observation Space (7-dimensional):
        - Interest rate at current period (normalized)
        - Interest rate at previous period
        - Interest rate at period t-2
        - Bond allocation ratio (0-1)
        - Equity allocation ratio (0-1)
        - Current period (0-40)
        - Funding ratio (assets/liabilities)
    
    Action Space (Discrete, 3 actions):
        - 0: Keep current asset allocation
        - 1: Increase bond allocation by 2%, decrease equity by 2%
        - 2: Decrease bond allocation by 2%, increase equity by 2%
    
    Reward:
        Change in funding ratio from previous step to current step,
        with negative rewards scaled by neg_multiple.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, sim_id=1, max_steps=40):
        """
        Initialize the LDI environment.
        
        Args:
            sim_id: Simulation ID (1-1000), determines which scenario to use
            max_steps: Maximum number of steps per episode (default: 40 quarters = 10 years)
        """
        super(LDIEnvironment, self).__init__()
        
        self.sim_id = sim_id
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        
        # Observation: 7-dimensional continuous space
        # [interest_rate_t, interest_rate_t-1, interest_rate_t-2, 
        #  bond_allocation, equity_allocation, period, funding_ratio]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 40.0, 10.0]),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.saa = None
        self.liabArray = None
        
    def getNewState(self, liabArray, dyn):
        """
        Get the current state representation.
        
        This method extracts the state from the simulation data, including
        interest rate history, asset allocation, and funding ratio.
        
        Args:
            liabArray: Current liability and asset array
            dyn: Current dynamic period
            
        Returns:
            numpy array of shape (7,) representing the current state
        """
        # Extract macro factor history for this simulation
        thmf = tmpArrays[(self.sim_id*(n_dyn+1)):(self.sim_id*(n_dyn+1)+n_dyn+1),:].copy()
        
        # Scale interest rate columns
        thmf[:,47] = thmf[:,47] * 100.0  # AA-rated bond yield
        thmf[:,46] = thmf[:,46] * 100.0  # 10-year treasury yield
        
        # Keep only the AA-rated bond yield column (index 8 after processing)
        keep_cols = [8]
        thmf = thmf[:,keep_cols]
        
        # Build state: current, previous, and t-2 interest rates
        newState = np.concatenate((
            thmf[dyn,:].flatten(), 
            thmf[max(0,dyn-1),:].flatten(), 
            thmf[max(dyn-2,0),:].flatten()
        ))
        
        # Add asset allocation ratios
        asset_alloc = liabArray[6:8]/np.sum(liabArray[6:8])
        newState = np.append(newState, asset_alloc)
        
        # Add current period
        newState = np.append(newState, dyn)
        
        # Add funding ratio (assets/liabilities)
        newState = np.append(newState, np.sum(liabArray[6:8])/liabArray[2])
        
        return newState.astype(np.float32)
    
    def get_reward(self, action, liabArray):
        """
        Calculate reward for the current action.
        
        The reward is the change in funding ratio after taking the action.
        Negative rewards are scaled to penalize funding ratio decreases more heavily.
        
        Args:
            action: Action taken (0=keep, 1=increase bonds, 2=decrease bonds)
            liabArray: Current liability and asset array
            
        Returns:
            tuple: (reward, updated_pboArray, new_saa)
        """
        assetArray = liabArray[6:8]
        
        # Determine new strategic asset allocation based on action
        if action == 0:  # keep current mix
            csaa = self.saa
        elif action == 1:  # increase bond investment
            csaa = np.minimum(1, np.maximum(0, self.saa + np.array([0.02,-0.02])))
        else:  # decrease bond investment (action == 2)
            csaa = np.minimum(1, np.maximum(0, self.saa + np.array([-0.02,0.02])))

        iq = self.current_step
        
        # Extract macro factors for this period
        thmf = tmpArrays[(self.sim_id*(n_dyn+1)):(self.sim_id*(n_dyn+1)+n_dyn+1),:].copy()
        thmf[:,47] = thmf[:,47] * 100.0
        thmf[:,46] = thmf[:,46] * 100.0
        keep_cols = [0,1,2,3,4,5,6,7,8,26,27,46,47]
        thmf = thmf[:,keep_cols]
        
        # Get projected benefit obligation (PBO) data
        pboArray = liabAll[((self.sim_id-1)*n_dyn+iq-1),0:4]
        liabNCF = pboArray[1] - pboArray[2]  # Net liability cash flow
        
        # Calculate returns for this period
        cashRtns = np.array([thmf[iq,11], thmf[iq,9]/4])  # Cash returns for bonds and equity
        priceRtns = np.array([thmf[iq,12], thmf[iq,10]])  # Price returns
        
        # Update asset values
        cashCF = np.sum(np.multiply(assetArray, cashRtns/100))
        assetArray = assetArray + np.multiply(assetArray, priceRtns/100)
        newAssetValue = np.sum(assetArray)
        
        # Rebalancing: adjust assets to match target allocation
        bsArray = np.multiply((cashCF+liabNCF+newAssetValue), csaa) - assetArray
        assetArray = assetArray + bsArray
        
        # Update PBO array with new values
        pboArray = np.insert(pboArray, 0, [self.sim_id, iq])
        pboArray = np.append(pboArray, assetArray)
        pboArray = np.append(pboArray, np.array([
            np.sum(assetArray),                    # Total asset value
            np.sum(assetArray)/pboArray[2],       # Funding ratio
            np.sum(assetArray)-pboArray[2]        # Funding surplus
        ]))
        
        # Calculate reward as change in funding ratio
        reward = - np.sum(liabArray[6:8])/liabArray[2] + np.sum(pboArray[6:8])/pboArray[2]
        
        # Scale negative rewards more heavily
        if reward < 0:
            reward = neg_multiple * reward
            
        return reward, pboArray, csaa
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., sim_id can be passed here)
            
        Returns:
            tuple: (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Allow changing sim_id through options
        if options and 'sim_id' in options:
            self.sim_id = options['sim_id']
        
        self.current_step = 0
        self.saa = [0.5, 0.5]  # Initial 50-50 allocation
        
        # Initialize liability and asset array
        self.liabArray = np.array([1, 0, planliab, 0, 0, 0])
        self.liabArray = np.append(self.liabArray, np.multiply(self.saa, plan_asset))
        self.liabArray = np.append(self.liabArray, np.array([
            plan_asset,
            plan_asset/planliab,
            plan_asset-planliab
        ]))
        
        observation = self.getNewState(self.liabArray, 0)
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (0, 1, or 2)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Get reward and update state
        reward, self.liabArray, self.saa = self.get_reward(action, self.liabArray)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observation
        if not terminated:
            observation = self.getNewState(self.liabArray, self.current_step)
        else:
            observation = self.getNewState(self.liabArray, self.current_step - 1)
        
        info = {
            'funding_ratio': self.liabArray[9],
            'total_assets': self.liabArray[8],
            'funding_surplus': self.liabArray[10],
            'saa': self.saa.copy(),
            'period': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment (optional for visualization).
        
        Args:
            mode: Rendering mode ('human' for text output)
        """
        if mode == 'human':
            print(f"Period: {self.current_step:2d} | "
                  f"SAA: [{self.saa[0]:.2f}, {self.saa[1]:.2f}] | "
                  f"Funding Ratio: {self.liabArray[9]:.4f} | "
                  f"Assets: ${self.liabArray[8]:,.0f}")
    
    def set_sim_id(self, sim_id):
        """
        Set the simulation ID for evaluation on different scenarios.
        
        Args:
            sim_id: Simulation ID (1-1000)
        """
        self.sim_id = sim_id

