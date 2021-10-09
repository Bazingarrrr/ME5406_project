from random import randint, choices



class Robot:
    def __init__(self) -> None:
        self.action_space = {
            3:'RIGHT',
            1:'DOWN',
            0:'UP',
            2:'LEFT',
            
        }
        self.position = [0, 0]

    def move(self, action, size):
        if action == 'UP':
            if self.position[0] - 1 >= 0:
                self.position[0] -= 1
        elif action == 'DOWN':
            if self.position[0] + 1 < size[0]:
                self.position[0] += 1
        elif action == 'LEFT':
            if self.position[1] - 1 >= 0:
                self.position[1] -= 1
        elif action == 'RIGHT':
            if self.position[1] + 1 < size[1]:
                self.position[1] += 1

    def choose_action(self, state:list, greedy:bool = True, policy=None):
        """
        state: [x_coordinate, y_coordinate]
        greedy: if greedy choose action greedily
                 if not greedy, choose action according to the weights
        """
        state = tuple(state)
        if policy == {} or policy is None: # choose action randomly
            return self.action_space[randint(0,3)]
        elif greedy: # choose action greedily
            return max( policy[state], key=policy[state].get ) 
        else:
            # choose action from policy
            weights = policy[state].values()
            actions = list(policy[state].keys())
            return choices(actions, weights=weights, k=1)[0]

            

