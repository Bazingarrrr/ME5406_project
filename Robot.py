from random import randint

class Robot:
    def __init__(self) -> None:
        self.action_space = {
            0:'UP',
            1:'DOWN',
            2:'LEFT',
            3:'RIGHT'
        }
        self.position = [0, 0]

    def choose_action(self, state):
        return self.action_space[randint(0,3)]


            

