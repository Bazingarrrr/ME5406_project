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
            episode = []
            num_steps = len(episode)

            for step in num_steps:
                # renew the returns, G
                # G = gama * G(t+1) + R(t+1), not sure whethere is t+1 or t

                # append G to returns(s, a)

                # get action greedly

                # Get new policy [policy improvement]
                self.action_space = ""
                pass
            

