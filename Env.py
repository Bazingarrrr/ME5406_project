from os import stat
import gym
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
        reward = self.get_reward()
        state = self.get_state()
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
        row = self.robot.position[0]
        column = self.robot.position[1]
        return self.map[row][column]

    def get_state(self):
        self.state = self.robot.position
        return self.state

    def print_log_info(self):
        print( "step: {0}, position of robot {1}".format(self.num_step ,self.state) )
        
    def monte_carlo_method(self):
        """
        # Monte Carlo Method:
        # 1. Sample and get the q(s,a) value
        # 2. Policy evaluation
        # 3. Policy improvement
        return: a policy pi(a|S)
        """
        # initialize parameter and delta-soft policy pi(a|S), 
        # as well as the discounted factor
        delta = 0.01
        gama = 0.5
        a = self.action_space
        pi = [0 for _ in range(len(self.action_space)) ]
        
        # dictionary of Q(s,a), R(s,a)
        Q = {}
        R = {}

        while True:
            # Generate a episode
            episode = [] # [[S, A, R],...]

            num_steps = len(episode)

            for step in num_steps:
                # renew the returns, G
                # G = gama * G(t+1) + R(t+1), not sure whethere is t+1 or t

                # append G to returns(s, a)

                # get action greedly

                # Get new policy [policy improvement]
                pass

    def run_an_episode(self):
        done = False
        state =  self.reset()
        episode = []
        while not done:  
            action = env.robot.choose_action(state)
            state, reward, done = env.step(action)
            episode.append( (state, action, reward) )
            env.print_log_info()
        return episode

if __name__ == "__main__":
    size = (4, 4)
    coord = [ (1,1), (1,3), (2,3), (3,0) ]
    env = Env(size, coord)
    episode = env.run_an_episode()
    pass



        
    


    

    