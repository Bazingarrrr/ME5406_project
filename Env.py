import gym
import pygame
from random import randint
from pygame import key

from pygame.time import delay
from Robot import Robot

LIGHTGRAY = (211, 211, 211)
TRAIN = ( 255, 182, 193 )
TEST = (255,255,255)

class Env(gym.Env):
    def __init__(self, size, coord=None, gui=False, use_memo=True, epsilon_decay=True) -> None:
        self.epsilon = 0.05
        self.gama = 0.9
        self.learning_rate = 1
        self.max_step = 100000

        self.map = None
        self.num_step = 0
        self.memo = []
        self.state = [0,0]

        self.use_memo = use_memo
        self.gui = gui
        self.epsilon_decay = epsilon_decay

        self.size = size
        self.coord = coord
        self.robot = Robot()

        if self.epsilon_decay is True:
            self.epsilon_max = 0.5
            self.epsilon_min = 0.0005
            self.epsilon_decay_value = 0.0005
        if self.gui:
            pygame.init()
            # Initializing surface
            self.background_color = (255, 255, 255)
            self.block_size = int( 200 * 4 / self.size[0] )
            self.screen_height = self.block_size * self.size[0]
            self.screen_width = self.block_size * self.size[1]
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
            self.screen.fill( self.background_color )

################################### Interaction #######################################
    def step(self, action):
        # renew the state
        self.robot.move(action, self.size)

        # get reward
        state = self.get_state()
        reward = self.get_reward()
        
        # update measurement parameters
        self.num_step += 1

        # check whether is done:
        # whether robot reaches the Frisbee
        done = False
        if self.map[state[0]][state[1]] != 0 or self.num_step > self.max_step:
            done = True
        return state, reward, done

    def reset(self):
        """
        reset the environment.
        """
        # build map
        if self.map is None:
            self.build_map()
            while self.check_map() is False:
                self.build_map()
        self.robot.position = [0,0]
        self.build_memo()
        # reset measure parameters
        self.num_step = 0

        return self.get_state()

    def get_reward(self):
        """
        discription about how to calculate reward.
        """
        penalty = extra_reward = 0
        row = self.robot.position[0]
        column = self.robot.position[1]
        self.memo[row][column] += 1
        if self.state == self.last_state:
            penalty += -1
        # punish dead-lock
        if self.memo[row][column] >= 2 and self.use_memo == True:
            penalty += -1
        if self.map[row][column] == 1:
            extra_reward = 0
        
        return 10*self.map[row][column] + penalty + extra_reward

    def get_state(self):
        self.last_state = self.state
        self.state = []
        self.state += self.robot.position
        return self.state

    def run_an_episode(self, policy=None):
        done = False
        state = self.reset()
        episode = []
        while not done:  
            action = self.robot.choose_action(state, greedy=True, policy=policy)
            next_state, reward, done = self.step(action)
            episode += [(state, action, reward)] 
            state = next_state
            # env.print_log_info()
        episode += [(state, '', 0)]
        return episode

    def reach_goal(self, state):
        return self.map[state[0]][state[1]] == 1

    def print_result(self):
        state = self.state
        if self.map[state[0]][state[1]] == 1:
            return "SUCCESS"
        elif self.map[state[0]][state[1]] == -1:
            return "FAIL"
        else:
            return "TIME OUT"
        
################################### Build MAP #######################################
    def build_map(self):
        self.build_map_without_hole()
        self.build_hole()
        if self.gui:
            self.draw_grid()

    def build_map_without_hole(self):
        size = self.size
        row = size[0]
        column = size[1]
        self.map = [[0 for i in range(column) ] for j in range(row) ]
        self.map[-1][-1] = 1

    def build_hole(self):
        if self.coord is None:
            num_of_holes = int( 0.25 * self.size[0] * self.size[1] )
            coords = set()
            for i in range(num_of_holes):
                 x = randint(0, self.size[0] - 1)
                 y = randint(0, self.size[1] - 1)
                 coords.add( (x,y) )
            coords = list(coords)
        else:
            coords = self.coord
        for coord in coords:
            row = coord[0]
            column = coord[1]
            self.map[row][column] = -1 
    
    def build_memo(self):
        self.memo = [[0 for i in range(self.size[1])] for j in range(self.size[0]) ]

    def check_map(self):
        """
        use DFS to check whether robot can reach the goal point.
        """
        # init
        print("checking map")
        position = [0, 0]
        used = [[ 0 for i in range(self.size[0]) ] for j in range(self.size[1])]

        # check origin
        if self.map[0][0] == -1 or self.map[-1][-1] == -1:
            return False
        return self.DFS_check_map(position, used)

    def DFS_check_map(self, position, used):
        # Whether reach the end point?
        if self.reach_goal(position):
            return True

        x, y = position[0], position[1]
        for action in self.robot.action_space.values():
            if action == 'UP':
                if position[0] - 1 >= 0 and self.map[position[0] -1][position[1]] != -1 :
                    position[0] -= 1
            elif action == 'DOWN' :
                if position[0] + 1 < self.size[0] and self.map[position[0]+1][position[1]] != -1 :
                    position[0] += 1
            elif action == 'LEFT' :
                if position[1] - 1 >= 0 and self.map[position[0]][position[1]-1] != -1 :
                    position[1] -= 1
            elif action == 'RIGHT' :
                if position[1] + 1 < self.size[1] and self.map[position[0]][position[1]+1] != -1 :
                    position[1] += 1

            next_x = position[0]
            next_y = position[1]
            if used[next_x][next_y] == 0:
                used[ next_x][ next_y ] = 1
                if self.DFS_check_map(position, used):
                    return True
                position[0], position[1] = x, y
            position[0], position[1] = x, y
        return False

################################### Training Method #######################################
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

                self.plot_result([episode[i-1]], background=TRAIN, delay_time=0)

                # renew Q(s,a)
                Q[state][action] = sum(returns[state][action]) / len( returns[state][action] )

                policy = self.derive_policy_from(Q, policy)
        return policy

    def SARSA(self, num_episode):
        """
        SARSA:
        """
        # initialize Q(s,a), all with arbitrary value but terminal state Q(s_terminal)=0
        Q, policy = self.init_Q_table()
        # self.epsilon = self.epsilon_max
        # Loop Begin
        while(num_episode):
            # epsilon decay
            if self.epsilon_decay and self.epsilon >= self.epsilon_min:
                self.epsilon -= self.epsilon_decay_value
            else:
                self.epsilon = 0

            # Initialize S
            state = self.reset()
            action = self.robot.choose_action(state, greedy=False, policy=policy)

            # Loop for each step of episode
            done = False
            while not done:
                # take action, observe reward and next_state
                next_state, reward, done = self.step(action)
                next_action = self.robot.choose_action(next_state, greedy=False, policy=policy) # SARSA - epsilon greedy
                
                # renew Q(s,a) 
                state = tuple(state)
                next_state = tuple(next_state)
                Q[state][action] = Q[state][action] + self.learning_rate * ( reward + self.gama * Q[next_state][next_action] - Q[state][action] )
                
                state = next_state
                action = next_action  
            policy = self.derive_policy_from(Q, policy)
            num_episode -= 1
        return policy

    def Qlearning(self, num_episode):
        """
        Qlearning
        """
        a = 1 # learning rate
        gama = self.gama

        Q, policy = self.init_Q_table()

        # initialize Q(s,a), all with arbitrary value but terminal state Q(s_terminal)=0

        # Loop Begin
        while(num_episode):
            # Initialize S
            state = self.reset()
            action = self.robot.choose_action(state, greedy=False, policy=policy)

            # Loop for each step of episode
            done = False
            while not done:
                # take action, observe reward and next_state
                next_state, reward, done = self.step(action)
                next_action = self.robot.choose_action(next_state, greedy=False, policy=policy)
                best_next_action = self.robot.choose_action(next_state, greedy=True, policy=policy)
                
                # renew Q(s,a) 
                state = tuple(state)
                next_state = tuple(next_state)
                Q[state][action] = Q[state][action] + a * ( reward + gama * Q[next_state][best_next_action] - Q[state][action] )
                state = next_state
                action = next_action
            num_episode -= 1
            policy = self.derive_policy_from(Q,policy)

        return policy

    def derive_policy_from(self, Q, policy):
        """
        Q: Q(s,a)
        """
        for state, _ in Q.items():
            best_action = max( Q[state], key=Q[state].get )
            for _, value in self.robot.action_space.items():
                policy[state][value] = self.epsilon/len(self.robot.action_space)
            policy[state][best_action] = 1 - self.epsilon + self.epsilon/len(self.robot.action_space)
        return policy

    def init_Q_table(self):
        """
        return a empty ( 0 value ) Q table
        as well as a policy.
        """
        Q = {}
        policy = {}
        self.reset()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                Q[(i,j)] = {}
                policy[(i,j)] = {}
                for action in self.robot.action_space.values():
                    policy[(i,j)][action] = self.epsilon/len(self.robot.action_space)
                    if self.map[i][j] == 0:
                        Q[(i,j)][action] = 0
                    else:
                        Q[(i,j)][action] = 0
        return Q, policy

################################### GUI #######################################
    def draw_grid(self, map=None):
        if map is None:
            map = self.map      
        for x in range(0, self.screen_width, self.block_size):
            for y in range(0, self.screen_height, self.block_size):
                j,i = int(x/self.block_size), int(y/self.block_size)
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                if map[i][j] == -1:
                    pygame.draw.rect(self.screen, (255,0,0), rect, 0)
                elif map[i][j] == 1:
                    pygame.draw.rect(self.screen, (0,255,0), rect, 0)
                else:
                    pygame.draw.rect(self.screen, self.background_color, rect, 0)
        self.draw_grid_line(map)
        pygame.display.flip()
        
    def get_circle_pos(self, state):
        x, y = state[0], state[1]
        x = 0.5 * self.block_size + x * self.block_size
        y = 0.5 * self.block_size + y * self.block_size
        return (y, x)

    def plot_result(self, episode, background=TEST,delay_time=0):
        """
        print map and robot at each step
        green - target point
        red - pitfall
        black - robot
        """
        self.background_color = background
        self.draw_grid()
        pos = tuple()
        for state, action, reward in episode:
            if pos != ():
                pygame.draw.circle(self.screen, self.background_color, pos, 0.9*self.block_size/2)
            pos = self.get_circle_pos(state)
            pygame.draw.circle(self.screen, (0,0,0), pos, 0.9*self.block_size/2)
            pygame.display.flip()
            delay(delay_time)
            

    def draw_grid_line(self, map=None):
        if map is None:
            map = self.map
        for x in range(0, self.screen_width, self.block_size):
            pygame.draw.line(self.screen, LIGHTGRAY, (x,0), (x,self.screen_height), 2)
        for y in range(0, self.screen_height, self.block_size):
            pygame.draw.line(self.screen, LIGHTGRAY, (0,y), (self.screen_width, y), 2)


if __name__ == "__main__":
    def run_test_10(test_num=10, policy_type='Qlearning'):
        size = (9,9)
        # coord = [ (1,1), (1,3), (2,3), (3,0) ]
        env = Env(size, gui=True)

        success_num = 0
        fail_num = 0
        timeout_num = 0
        for i in range(test_num):
            print("Begin test for round {0}".format(i))
            env.map = None
            if policy_type == 'qlearning':
                policy = env.Qlearning(2000)
            elif policy_type == 'SARSA':
                policy = env.SARSA(2000)
            elif policy_type == 'monte_carlo':
                policy = policy = env.monte_carlo_method(2000)
            episode = env.run_an_episode(policy=policy)
            if env.print_result() == 'SUCCESS':
                success_num += 1
                print('SUCCESS')
            elif env.print_result() == 'FAIL':
                fail_num += 1
                print('FAIL')
            else:
                timeout_num += 1
                print('TIME OUT')
            if env.gui:
                env.plot_result(episode,delay_time=30)
        print("success rate {0}%, fail rate {1}%, time out rate {2}%".format( 100*success_num/test_num, 
                                                                            100*fail_num/test_num,
                                                                            100*timeout_num/test_num ))

    def run_test_4(test_num=10):
        size = (4, 4)
        coord = [ (1,1), (1,3), (2,3), (3,0) ]
        env = Env(size, gui=True, coord=coord, use_memo=True)

        success_num = 0
        fail_num = 0
        timeout_num = 0
        for i in range(test_num):
            print("Begin test for round {0}".format(i))
            policy = env.monte_carlo_method(2000)
            # policy = env.SARSA(2000)
            # policy = env.Qlearning(2000)
            episode = env.run_an_episode(policy=policy)
            if env.print_result() == 'SUCCESS':
                success_num += 1
                print('SUCCESS')
            elif env.print_result() == 'FAIL':
                fail_num += 1
                print('FAIL')
            else:
                timeout_num += 1
                print('TIME OUT')
            if env.gui:
                env.plot_result(episode,delay_time=500)
        print("success rate {0}%, fail rate {1}%, time out rate {2}%".format( 100*success_num/test_num, 
                                                                            100*fail_num/test_num,
                                                                            100*timeout_num/test_num ))

    # run_test_4(test_num=10)
    # run_test_10(test_num=50, policy_type='SARSA')
    # run_test_10(test_num=20, policy_type='qlearning')
    run_test_10(test_num=20, policy_type='monte_carlo')
        
    


    

    