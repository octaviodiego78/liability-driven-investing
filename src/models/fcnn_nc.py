"""
FCNN model without constraints implementation.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from config import (MacroFactor, VARModel, Mapping, Mortality, Census, RecFun,
                     histMF, histAR, cholMF, cholNormal, cholRecession, termmix,
                     migration, batch_size, gamma, eps_start, eps_end, eps_decay,
                     static_update, neg_multiple, n_dyn, plan_asset, planliab, employees, var, fundmap, tmpArrays, liabAll, multiple)
from models.base_model import FCNNDQN, ReplayMemory, Tensor, Transition
import random
import math

class FCNNWithoutConstraint:
    """
    Fully Connected Neural Network without constraints for LDI.
    Implements the unconstrained version of the FCNN model.
    """
    
    def __init__(self):
        """Initialize the FCNN model without constraints."""
        self.model = FCNNDQN(n_actions=51)  # 51 possible allocation combinations
        self.static_model = FCNNDQN(n_actions=51)
        self.static_model.load_state_dict(self.model.state_dict())
        self.static_model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(400000)
        self.steps_done = 0
        
    def getNewState(self, employees, var, fundmap, histMF, histAR, cholMF, cholNormal, cholRecession,
                  liabArray, tmpArray, sim, dyn):
        """Get the current state representation."""
        iq = dyn
        thmf = tmpArray[(sim*(n_dyn+1)):(sim*(n_dyn+1)+n_dyn+1),:].copy()
        thmf[:,47] = thmf[:,47] * 100.0
        thmf[:,46] = thmf[:,46] * 100.0
        keep_cols = [8]
        thmf = thmf[:,keep_cols]
        newState = np.concatenate((thmf[dyn,:].flatten(), thmf[max(0,dyn-1),:].flatten(), thmf[max(dyn-2,0),:].flatten()))
        asset_alloc = liabArray[6:8]/np.sum(liabArray[6:8])
        newState = np.append(newState, asset_alloc)
        newState = np.append(newState, dyn)
        newState = np.append(newState, np.sum(liabArray[6:8])/liabArray[2])
        return newState
        
    def get_reward(self, employees, saa, action, var, fundmap, histMF, histAR, cholMF, cholNormal, cholRecession,
                   liabArray, tmpArray, liabAll, sim, dyn, multiple, planliab=planliab, bs=True,
                   inter="linear", extro="tar", target=0.04):
        """Calculate reward for the current action."""
        assetArray = liabArray[6:8]
        # Different from constrained version - direct allocation based on action
        csaa = np.array([0.02*action.item(), 1-0.02*action.item()])

        iq = dyn
        thmf = tmpArray[(sim*(n_dyn+1)):(sim*(n_dyn+1)+n_dyn+1),:].copy()
        thmf[:,47] = thmf[:,47] * 100.0
        thmf[:,46] = thmf[:,46] * 100.0
        keep_cols = [0,1,2,3,4,5,6,7,8,26,27,46,47]
        thmf = thmf[:,keep_cols]
        pboArray = liabAll[((sim-1)*n_dyn+iq-1),0:4]
        liabNCF = pboArray[1] - pboArray[2]
        cashRtns = np.array([thmf[iq,11],thmf[iq,9]/4])
        priceRtns = np.array([thmf[iq,12],thmf[iq,10]])
        cashCF = np.sum(np.multiply(assetArray,cashRtns/100))
        assetArray = assetArray+np.multiply(assetArray,priceRtns/100)
        newAssetValue = np.sum(assetArray)
        
        if bs == False and (cashCF+liabNCF) > 0:
            bsArray = np.multiply((cashCF+liabNCF),csaa)
        elif bs == False and (cashCF + liabNCF) <= 0:
            bsArray = np.multiply((cashCF+liabNCF),csaa)
        else:
            bsArray = np.multiply((cashCF+liabNCF+newAssetValue),csaa)-assetArray
            
        assetArray = assetArray + bsArray
        pboArray = np.insert(pboArray,0,[sim,iq])
        pboArray = np.append(pboArray,assetArray)
        pboArray = np.append(pboArray,np.array([np.sum(assetArray),np.sum(assetArray)/pboArray[2],np.sum(assetArray)-pboArray[2]]))
        reward = - np.sum(liabArray[6:8])/liabArray[2] + np.sum(pboArray[6:8])/pboArray[2]
        if reward < 0:
            reward = neg_multiple*reward
        return reward, pboArray, csaa
        
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state.type(Tensor)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(51)]], dtype=torch.long)
            
    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < batch_size:
            return 0.0, 0.0
            
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch.type(Tensor)).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size).type(Tensor)
        next_state_values[non_final_mask] = self.static_model(non_final_next_states.type(Tensor)).max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return float(loss), float(torch.mean(expected_state_action_values))
        
    def train(self, num_episodes, num_sims):
        """Train the model."""
        for i_episode in range(num_episodes):
            saa = [0.5, 0.5]
            liabArray = np.array([1,0,planliab,0,0,0])
            liabArray = np.append(liabArray, np.multiply(saa, plan_asset))
            liabArray = np.append(liabArray, np.array([plan_asset,
                                                      plan_asset/planliab,
                                                      plan_asset-planliab]))
            
            if (i_episode+1) % num_sims == 0:
                isim = num_sims
            else:
                isim = (i_episode+1) % num_sims
                
            state = self.getNewState(employees, var, fundmap, histMF, histAR, cholMF, cholNormal, cholRecession,
                                 liabArray, tmpArrays, isim, 0)
            newInput = np.copy(liabArray)
            state = torch.from_numpy(state).view(1, 7)
            
            for idyn in range(1, 41):
                action = self.select_action(state)
                reward, newInput, saa = self.get_reward(employees, saa, action, var, fundmap, histMF, histAR,
                                                      cholMF, cholNormal, cholRecession, newInput, tmpArrays,
                                                      liabAll, isim, idyn, multiple, planliab)
                
                reward = Tensor([reward])
                if idyn == 40:
                    next_state = None
                else:
                    next_state = self.getNewState(employees, var, fundmap, histMF, histAR, cholMF, cholNormal,
                                              cholRecession, newInput, tmpArrays, isim, idyn)
                    next_state = torch.from_numpy(next_state).view(1, 7)
                
                self.memory.push(state, action, next_state, reward)
                state = next_state
                loss, val = self.optimize_model()
                
            if i_episode % 10 == 9:
                print(f"Episode: {i_episode+1}")
            if i_episode % static_update == 0:
                self.static_model.load_state_dict(self.model.state_dict())
                
    def evaluate(self, num_sims):
        """Evaluate the trained model."""
        # Initialize arrays to store results
        tsaa = [0.5, 0.5]
        tliabArray = np.array([1, 0, planliab, 0, 0, 0])
        tliabArray = np.append(tliabArray, np.multiply([0.5, 0.5], plan_asset))
        tliabArray = np.append(tliabArray, np.array([plan_asset, plan_asset/planliab, plan_asset-planliab]))
        tliabArray = np.copy(tliabArray)
        
        sims = []
        sims_sim = []
        qs = []
        rwds = []
        rwds_total = []
        
        for i_sim in range(1, num_sims+1):
            # Initialize the environment and state
            saa = [0.5, 0.5]
            liabArray = np.array([1, 0, planliab, 0, 0, 0])
            liabArray = np.append(liabArray, np.multiply(saa, plan_asset))
            liabArray = np.append(liabArray, np.array([plan_asset,
                                                      plan_asset/planliab,
                                                      plan_asset-planliab]))
            
            state = self.getNewState(employees, var, fundmap, histMF, histAR, cholMF, cholNormal, cholRecession,
                                 liabArray, tmpArrays, i_sim, 0)
            newInput = np.copy(liabArray)
            state = torch.from_numpy(state).view(1, 7)
            reward_total = 0
            
            for idyn in range(1, 41):
                # Select and perform an action (best action, no exploration)
                with torch.no_grad():
                    action = self.model(state.type(Tensor)).max(1)[1].view(1, 1)
                    
                reward, newInput, saa = self.get_reward(employees, saa, action, var, fundmap, histMF, histAR,
                                                      cholMF, cholNormal, cholRecession, newInput, tmpArrays,
                                                      liabAll, i_sim, idyn, multiple, planliab,
                                                      bs=True, inter="linear", extro="tar", target=0.04)
                
                reward_total = reward_total + reward * gamma**(idyn-1)
                
                # Accumulate results
                tsaa = np.vstack([tsaa, np.array(saa)])
                tliabArray = np.vstack([tliabArray, np.array(newInput)])
                
                reward = Tensor([reward])
                
                if idyn == 40:
                    next_state = None
                else:
                    next_state = self.getNewState(employees, var, fundmap, histMF, histAR, cholMF, cholNormal,
                                              cholRecession, newInput, tmpArrays, i_sim, idyn)
                    next_state = torch.from_numpy(next_state).view(1, 7)
                state = next_state
                
                # Store tracking data
                sims.append(i_sim + 1)
                qs.append(idyn + 1)
                rwds.append(float(reward))
                
            rwds_total.append(reward_total)
            sims_sim.append(i_sim)
            
            if i_sim % 50 == 49:
                print(f"Simulation: {i_sim+1}")
                
        return tliabArray, tsaa, sims, qs, rwds, rwds_total, sims_sim