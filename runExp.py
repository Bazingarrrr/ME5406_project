from os import set_inheritable
import Env
import multiprocessing

def run_test(size=(10,10), test_num=10, episode_num=250):
    """
    
    """
    ################################  Multi-Training Test  ########################################
    settings = [ (False, 'monte_carlo'),
             (False ,'SARSA'), 
             (False,'qlearning')]

    # Start Process for Multi-Training Test of each method
    # Single-Training Test will run on the last round
    process_list = []
    for if_gui, policy in settings:
        p = multiprocessing.Process(target=Env.run_test, args=(size, test_num, if_gui, policy, episode_num) )
        p.start()
        process_list.append(p)

    # Wait each process to end
    for p in process_list:
        p.join()
    ################################  GUI Demonstration ########################################
    Env.single_test(size, if_gui=True, policy_type='monte_carlo', episode_num=30)
    return 

if __name__ == '__main__':
    settings = [ ((4,4),500), 
              ((10,10), 10000)]
    multiprocessing.set_start_method('spawn') # used to handle multi-processsing problem
    for size, episode_num in settings:
        run_test(size=size, test_num=50, episode_num=episode_num)
    # Env.run_test((4,4), test_num=1, if_gui=False, policy_type='monte_carlo', episode_num=20)
