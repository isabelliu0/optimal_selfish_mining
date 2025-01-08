import numpy as np
from scipy import sparse
import mdptoolbox
from dataclasses import dataclass
from enum import IntEnum

class Fork(IntEnum):
    """Represents the fork state in the blockchain"""
    IRRELEVANT = 0  # Match not feasible, last block is selfish OR honest branch empty
    RELEVANT = 1    # Match feasible if a>=h, e.g. last block is honest 
    ACTIVE = 2      # Just performed a match

class Action(IntEnum):
    """Possible actions in the mining strategy"""
    ADOPT = 0
    OVERRIDE = 1 
    MATCH = 2
    WAIT = 3

@dataclass
class State:
    """Represents a state in the mining strategy"""
    a: int  # Attacker's chain length
    h: int  # Honest chain length
    fork: Fork

class SelfishMining:
    def __init__(self, alpha_power: float, gamma_ratio: float, max_fork_len: int):
        """
        Initialize the selfish mining strategy solver
        
        Args:
            alpha_power: Mining power share of the attacker
            gamma_ratio: Proportion of honest miners that would mine on attacker's chain during tie
            max_fork_len: Maximum length of a block fork
        """
        self.alpha_power = alpha_power
        self.gamma_ratio = gamma_ratio
        self.max_fork_len = max_fork_len
        
        # Calculate total number of states
        self.num_states = (max_fork_len + 1) * (max_fork_len + 1) * 3
        print(f"Number of states: {self.num_states}")
        
        # Initialize transition and reward matrices using LIL format for efficient construction
        num_actions = len(Action)
        self.P = [sparse.lil_matrix((self.num_states, self.num_states)) for _ in range(num_actions)]
        self.Rs = [sparse.lil_matrix((self.num_states, self.num_states)) for _ in range(num_actions)]
        self.Rh = [sparse.lil_matrix((self.num_states, self.num_states)) for _ in range(num_actions)]
        self.W_rou = [sparse.lil_matrix((self.num_states, self.num_states)) for _ in range(num_actions)]
        
        # Build the matrices
        self._build_matrices()

    def st2stnum(self, state: State) -> int:
        """Convert state to state number"""
        if state.a > self.max_fork_len or state.h > self.max_fork_len:
            raise ValueError("Block fork is too long")
        return state.a * (self.max_fork_len + 1) * 3 + state.h * 3 + state.fork

    def stnum2st(self, num: int) -> State:
        """Convert state number to state"""
        num = num
        fork = Fork(num % 3)
        temp = num // 3
        h = temp % (self.max_fork_len + 1)
        a = temp // (self.max_fork_len + 1)
        return State(a, h, fork)

    def _build_matrices(self):
        """Build transition and reward matrices"""
        # Define adopt transitions
        st_1_0 = self.st2stnum(State(1, 0, Fork.IRRELEVANT))
        st_0_1 = self.st2stnum(State(0, 1, Fork.IRRELEVANT))
        
        # Set adopt transition probabilities
        for i in range(self.num_states):
            self.P[Action.ADOPT][i, st_1_0] = self.alpha_power
            self.P[Action.ADOPT][i, st_0_1] = 1 - self.alpha_power

        for i in range(self.num_states):
            if i % 2000 == 0:
                print(f"Processing state: {i}")

            state = self.stnum2st(i)
            
            # Set adopt rewards
            self.Rh[Action.ADOPT][i, st_1_0] = state.h
            self.Rh[Action.ADOPT][i, st_0_1] = state.h

            # Define override
            if state.a > state.h:
                st_next1 = self.st2stnum(State(state.a - state.h, 0, Fork.IRRELEVANT))
                st_next2 = self.st2stnum(State(state.a - state.h - 1, 1, Fork.RELEVANT))
                
                self.P[Action.OVERRIDE][i, st_next1] = self.alpha_power
                self.Rs[Action.OVERRIDE][i, st_next1] = state.h + 1
                
                self.P[Action.OVERRIDE][i, st_next2] = 1 - self.alpha_power
                self.Rs[Action.OVERRIDE][i, st_next2] = state.h + 1
            else:
                self.P[Action.OVERRIDE][i, 0] = 1
                self.Rh[Action.OVERRIDE][i, 0] = 10000

            # Define wait
            if (state.fork != Fork.ACTIVE and 
                state.a + 1 <= self.max_fork_len and 
                state.h + 1 <= self.max_fork_len):
                
                st_next1 = self.st2stnum(State(state.a + 1, state.h, Fork.IRRELEVANT))
                st_next2 = self.st2stnum(State(state.a, state.h + 1, Fork.RELEVANT))
                
                self.P[Action.WAIT][i, st_next1] = self.alpha_power
                self.P[Action.WAIT][i, st_next2] = 1 - self.alpha_power
                
            elif (state.fork == Fork.ACTIVE and 
                  state.a > state.h and 
                  state.h > 0 and 
                  state.a + 1 <= self.max_fork_len and 
                  state.h + 1 <= self.max_fork_len):
                
                st_next1 = self.st2stnum(State(state.a + 1, state.h, Fork.ACTIVE))
                st_next2 = self.st2stnum(State(state.a - state.h, 1, Fork.RELEVANT))
                st_next3 = self.st2stnum(State(state.a, state.h + 1, Fork.RELEVANT))
                
                self.P[Action.WAIT][i, st_next1] = self.alpha_power
                self.P[Action.WAIT][i, st_next2] = self.gamma_ratio * (1 - self.alpha_power)
                self.Rs[Action.WAIT][i, st_next2] = state.h
                self.P[Action.WAIT][i, st_next3] = (1 - self.gamma_ratio) * (1 - self.alpha_power)
            else:
                self.P[Action.WAIT][i, 0] = 1
                self.Rh[Action.WAIT][i, 0] = 10000

            # Define match
            if (state.fork == Fork.RELEVANT and 
                state.a >= state.h and 
                state.h > 0 and 
                state.a + 1 <= self.max_fork_len and 
                state.h + 1 <= self.max_fork_len):
                
                st_next1 = self.st2stnum(State(state.a + 1, state.h, Fork.ACTIVE))
                st_next2 = self.st2stnum(State(state.a - state.h, 1, Fork.RELEVANT))
                st_next3 = self.st2stnum(State(state.a, state.h + 1, Fork.RELEVANT))
                
                self.P[Action.MATCH][i, st_next1] = self.alpha_power
                self.P[Action.MATCH][i, st_next2] = self.gamma_ratio * (1 - self.alpha_power)
                self.Rs[Action.MATCH][i, st_next2] = state.h
                self.P[Action.MATCH][i, st_next3] = (1 - self.gamma_ratio) * (1 - self.alpha_power)
            else:
                self.P[Action.MATCH][i, 0] = 1
                self.Rh[Action.MATCH][i, 0] = 10000

        # Convert matrices to CSR format for efficiency
        for i in range(len(Action)):
            self.P[i] = self.P[i].tocsr()
            self.Rs[i] = self.Rs[i].tocsr()
            self.Rh[i] = self.Rh[i].tocsr()

    def mdp_relative_value_iteration(self, P, R, epsilon=0.01):
        """
        Implement relative value iteration for MDPs
        """
        P = [p.toarray() for p in P]  # Convert sparse matrices to numpy arrays
        R = [r.toarray() for r in R]
        
        # Create MDP object
        mdp = mdptoolbox.mdp.RelativeValueIteration(
            transitions=P, 
            reward=R,
            epsilon=epsilon,
        )
        
        # Solve MDP
        mdp.run()
        
        return mdp.policy, mdp.average_reward

    def solve_strategy(self, epsilon: float = 0.0001):
        """
        Solve for optimal selfish mining strategy
        """
        # Binary search for lower bound
        low_rou = 0
        high_rou = 1
        
        while high_rou - low_rou > epsilon/8:
            rou = (high_rou + low_rou) / 2
            
            # Update W_rou matrices
            for i in range(len(Action)):
                self.W_rou[i] = (1-rou) * self.Rs[i] - rou * self.Rh[i]
            
            policy, reward = self.mdp_relative_value_iteration(self.P, self.W_rou, epsilon/8)
            
            if reward > 0:
                low_rou = rou
            else:
                high_rou = rou
                
        lower_bound_rou = rou
        print(f"Lower bound reward: {rou}")
        
        # Binary search for upper bound
        low_rou = rou
        high_rou = min(rou + 0.1, 1)
        
        while high_rou - low_rou > epsilon/8:
            rou = (high_rou + low_rou) / 2
            
            # Update rewards for boundary states
            self._update_boundary_rewards(rou)
            
            # Update W_rou matrices
            for i in range(len(Action)):
                self.W_rou[i] = (1-rou) * self.Rs[i] - rou * self.Rh[i]
            
            rou_prime = max(low_rou - epsilon/4, 0)
            policy, reward, _ = self.mdp_relative_value_iteration(self.P, self.W_rou, epsilon/8)
            
            if reward > 0:
                low_rou = rou
            else:
                high_rou = rou
                
        print(f"Upper bound reward: {rou}")
        return lower_bound_rou, rou

    def _update_boundary_rewards(self, rou: float):
        """Update rewards for boundary states"""
        for i in range(self.num_states):
            state = self.stnum2st(i)
            if state.a == self.max_fork_len:
                mid1 = ((1-rou) * self.alpha_power * (1-self.alpha_power) / 
                       (1-2*self.alpha_power)**2 + 
                       0.5*((state.a-state.h)/(1-2*self.alpha_power) + state.a + state.h))
                
                st_next1 = self.st2stnum(State(1, 0, Fork.IRRELEVANT))
                st_next2 = self.st2stnum(State(0, 1, Fork.IRRELEVANT))
                
                self.Rs[Action.ADOPT][i, st_next1] = mid1
                self.Rs[Action.ADOPT][i, st_next2] = mid1
                self.Rh[Action.ADOPT][i, st_next1] = 0
                self.Rh[Action.ADOPT][i, st_next2] = 0
                
            elif state.h == self.max_fork_len:
                mid1 = self.alpha_power * (1-self.alpha_power) / ((1-2*self.alpha_power)**2)
                mid2 = (self.alpha_power/(1-self.alpha_power))**(state.h-state.a)
                mid3 = ((1-mid2)*(0-rou)*state.h + 
                       mid2*(1-rou)*(mid1+(state.h-state.a)/(1-2*self.alpha_power)))
                
                st_next1 = self.st2stnum(State(1, 0, Fork.IRRELEVANT))
                st_next2 = self.st2stnum(State(0, 1, Fork.IRRELEVANT))
                
                self.Rs[Action.ADOPT][i, st_next1] = mid3
                self.Rs[Action.ADOPT][i, st_next2] = mid3
                self.Rh[Action.ADOPT][i, st_next1] = 0
                self.Rh[Action.ADOPT][i, st_next2] = 0

def main():
    alpha_power = 0.475
    gamma_ratio = 0
    max_fork_len = 160 if alpha_power >= 0.45 else 80
    
    print(f"Running with alpha_power={alpha_power}, gamma_ratio={gamma_ratio}, max_fork_len={max_fork_len}")
    
    solver = SelfishMining(alpha_power, gamma_ratio, max_fork_len)
    lower_bound, upper_bound = solver.solve_strategy(epsilon=0.0001)
    print(f"Solution bounds: [{lower_bound}, {upper_bound}]")

if __name__ == "__main__":
    main()