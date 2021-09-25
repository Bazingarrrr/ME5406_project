import gym
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

        # update measurement parameters
        self.num_step += 1

        # check whether is done:
        # whether robot reaches the Frisbee
        done = False
        if self.get_state() == [ size[0]-1, size[1]-1 ]:
            done = True
        
        return done


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
        pass

    def get_state(self):
        self.state = self.robot.position
        return self.state

    def print_log_info(self):
        print( "step: {0}, position of robot {1}".format(self.num_step ,self.state) )
        
        
if __name__ == "__main__":
    size = (4, 4)
    coord = [ (1,1), (1,3), (2,3), (3,0) ]
    env = Env(size, coord)

    done = False
    state =  env.reset()
    while not done:
        state = env.get_state()
        action = env.robot.choose_action(state)
        done = env.step(action)
        env.print_log_info()


        
    


    

    