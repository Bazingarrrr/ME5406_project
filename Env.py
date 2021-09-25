from os import stat
import gym
from numpy.lib.function_base import average
from numpy.lib.shape_base import column_stack
from Robot import Robot



class Env(gym.Env):
    def __init__(self, size, coord) -> None:
        self.map = []
        self.size = size
        self.coord = coord
        self.robot = Robot()
        self.num_step = 0
        self.state = [0,0]


    def step(self, action):
        # renew the state
        if action == 'UP':
            if self.robot.position[0] - 1 >= 0:
                self.robot.position[0] -= 1
        elif action == 'DOWN':
            if self.robot.position[0] + 1 < self.size[0]:
                self.robot.position[0] += 1
        elif action == 'LEFT':
            if self.robot.position[1] - 1 >= 0:
                self.robot.position[1] -= 1
        elif action == 'RIGHT':
            if self.robot.position[1] + 1 < self.size[1]:
                self.robot.position[1] += 1
        
        # get reward
        state = self.get_state()
        reward = self.get_reward()
        
        # update measurement parameters
        self.num_step += 1

        # check whether is done:
        # whether robot reaches the Frisbee
        done = False
        if state == [ self.size[0]-1, self.size[1]-1 ]:
            done = True
        
        return state, reward, done


    def reset(self):
        """
        reset the environment.
        """
        # build map
        self.build_map()
        self.build_hole()

        self.robot.position = [0,0]
        
        # reset measure parameters
        self.num_step = 0

        return self.get_state()

    def build_map(self):
        size = self.size
        row = size[0]
        column = size[1]
        self.map = [[0 for i in range(column) ] for j in range(row) ]
        self.map[-1][-1] = 1

    def build_hole(self):
        coords = self.coord
        for coord in coords:
            row = coord[0]
            column = coord[1]
            self.map[row][column] = -1 
    
    def get_reward(self):
        """
        discription about how to calculate reward.
        """
        penalty = 0
        if self.state == self.last_state:
            penalty = -100
        row = self.robot.position[0]
        column = self.robot.position[1]
        return self.map[row][column] + penalty

    def get_state(self):
        self.last_state = self.state
        self.state = []
        self.state += self.robot.position
        return self.state

    def print_log_info(self):
        print( "step: {0}, position of robot {1}".format(self.num_step ,self.state) )
        
    def monte_carlo_method(self, num_episode):
        """
        # Monte Carlo Method:
        # 1. Sample and get the q(s,a) value
        # 2. Policy evaluation
        # 3. Policy improvement
        return: a policy pi(a|S)
        """
        # initialize parameter and delta-soft policy pi(a|S), 
        # as well as the discounted factor
        delta = 0.05
        gama = 0.9
        policy = {} # pi(a|s)
        
        # dictionary of Q(s,a), return(s,a)
        Q = {}
        returns = {}

        while (num_episode):
            # Generate a episode
            episode = self.run_an_episode() # [[S(t), A(t), R(t+1)],...]
            num_episode -= 1
            num_steps = len(episode)

            # initialize G
            G = 0
            for i in range(1, num_steps+1):
                state = episode[-i][0]
                action = episode[-i][1]
                reward = episode[-i][2]

                # renew the returns, G
                G = gama * G + reward

                # append G to returns(s, a)
                state = tuple(state)
                if state not in returns:
                    returns[state] = {}
                    Q[state] = {}
                    policy[state] = {}
                if action not in returns[state]:
                    returns[state][action] = []
                    Q[state][action] = 0
                returns[state][action].append(G)

                # renew Q(s,a)
                Q[state][action] = sum(returns[state][action]) / len( returns[state][action] )

                # choose action greedily
                best_action = max( Q[state], key=Q[state].get )

                # Get new policy [policy improvement]
                for _, value in self.robot.action_space.items():
                    policy[state][value] = delta/len(self.robot.action_space)
                policy[state][best_action] = 1 - delta + delta/len(self.robot.action_space)
        
        return policy


    def run_an_episode(self):
        done = False
        state =  self.reset()
        episode = []
        while not done:  
            action = env.robot.choose_action(state)
            state, reward, done = env.step(action)
            episode += [(state, action, reward)] 
            env.print_log_info()
        return episode

if __name__ == "__main__":
    size = (4, 4)
    coord = [ (1,1), (1,3), (2,3), (3,0) ]
    env = Env(size, coord)
    policy = env.monte_carlo_method(1000)
    print(policy)



        
    


    

    