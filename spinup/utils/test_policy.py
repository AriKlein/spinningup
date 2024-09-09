import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import pandas as pd
import numpy as np

def taxi_decode(i):
    dest_idx = 3
    pass_loc = 4
    #pass_loc = 4*(i%2)
    #i = i//2
    col_idx = i%5
    i = i//5
    row_idx = i%5
    return [row_idx, col_idx, 4, 3] # pass_loc,dest_idx]

#ARI:  decode to make the passenger and destination location have the same units
def taxi_decode_v2(i):
    (taxi_row, taxi_col, pass_loc, dest_idx) = list(taxi_decode(i))
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    dest_row = locs[dest_idx][0]
    dest_col = locs[dest_idx][1]
    pass_in_taxi = 0
    if pass_loc==4:
        pass_row = taxi_row
        pass_col = taxi_col
        pass_in_taxi = 1
    else:
        pass_row = locs[pass_loc][0]
        pass_col = locs[pass_loc][1]
        pass_in_taxi = 0
    return list(taxi_decode(i)) # (taxi_row, taxi_col, dest_row, dest_col)  #  (taxi_row, taxi_col, pass_row, pass_col, dest_row, dest_col, pass_in_taxi)



'''
def taxi_decode(i):
    dest_idx = 3
    pass_loc = 4
    #pass_loc = 4*(i%2)
    #i = i//2
    col_idx = i%5
    i = i//5
    row_idx = i%5
    return [row_idx, col_idx, pass_loc,dest_idx]

#ARI:  decode to make the passenger and destination location have the same units
def taxi_decode_v2(i):
    (taxi_row, taxi_col, pass_loc, dest_idx) = list(taxi_decode(i))
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    dest_row = locs[dest_idx][0]
    dest_col = locs[dest_idx][1]
    pass_in_taxi = 0
    if pass_loc==4:
        pass_row = taxi_row
        pass_col = taxi_col
        pass_in_taxi = 1
    else:
        pass_row = locs[pass_loc][0]
        pass_col = locs[pass_loc][1]
        pass_in_taxi = 0
    return (taxi_row, taxi_col, dest_row, dest_col)  #  (taxi_row, taxi_col, pass_row, pass_col, dest_row, dest_col, pass_in_taxi)
'''

def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        o = list(taxi_decode_v2(o)) # ARI:  hack to make this work for taxi problem
        a = get_action(o)
        taxi_action = int(a)
        o, r, d, _ = env.step(taxi_action) # env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()



# Method to pretty-print the policies:
# - Prints a policy map for each passenger location given that the passenger is not yet in the taxi, so the taxi should be trying to get to and pick up the passenger
# - Prints a policy map for each destination location given that the passenger is already in the taxi, so the taxi should be trying to get to the destination and drop off the passenger
def pretty_print_policy(taxi, local_policy):

    MAP = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ]

    direction_repr = {1:' 🡑 ', 2:' 🡒 ', 3:' 🡐 ', 0:' 🡓 ', 4:' + ', 5:' - ', None:' ⬤ '}

    # Print policies for states where we are trying to get to passenger, so dest_idx is irrelevant, as long as not = pass_idx
    '''
    print('Passenger not in taxi, pass at Red (top left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 0, 1)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at Green (Top Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 1, 0)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at yellow (Bottom Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 2, 0)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at Blue (Bottom Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 3, 0)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()



    # Print policies for states where we already have passenger and are trying to get to destination, so pass_idx is always 4

    print('Passenger in taxi, Dest = Red (Top Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 0)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger in taxi, Dest = Green (Top Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 1)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger in taxi, Dest = Yellow (Bottom Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 2)
            print(taxi.MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()
    '''

    print('Passenger in taxi, Dest = Blue (Bottom Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 3)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

def export_policy_to_excel(fpath, itr, deterministic=False):

    fname = osp.join(fpath, 'pyt_save','model.pt') # 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    possible_actions = ["South", "North", "East", "West", "Pickup/Dropoff"]

    excel_df = {}
    excel_df_keys_state = ['Taxi Row', 'Taxi Column', 'Destination Row', 'Destination Column']
    excel_df_keys_actions = []
    for ii in possible_actions:
        excel_df_keys_actions.append(['pi('+ii+')'])
    # excel_df_keys_actions.append = excel_df_keys + ['V']

    my_policy = np.zeros(env.nS)

    for ii in range(len(excel_df_keys_state)):
        excel_df[excel_df_keys_state[ii]] = []

    for ii in range(len(excel_df_keys_actions)):
        excel_df[excel_df_keys_actions[ii][0]] = []

    excel_df['Sum Probs'] = []
    excel_df['V'] = []
    excel_df['Highest Probability Action'] = []


    for o in range(env.nS):
        state_index = o
        # Decode state and put into Excel DF
        o = list(taxi_decode_v2(o))
        for ii in range(len(o)):
            excel_df[excel_df_keys_state[ii]].append(o[ii])

        # Pass state to model
        o = torch.as_tensor(o, dtype=torch.float32)
        a, v, logp = model.step(o)
        p = model.ari_get_distribution(o).probs # np.exp(logp)

        for ii in range(len(p)):
            excel_df[excel_df_keys_actions[ii][0]].append(float(p[ii]))

        excel_df['Sum Probs'].append(float(sum(p)))
        excel_df['V'].append(float(v))
        excel_df['Highest Probability Action'].append(possible_actions[int(np.argmax(p))])

        my_policy[state_index] = int(np.argmax(p))

    df = pd.DataFrame(excel_df)

    writer = pd.ExcelWriter('Simple_Taxi_Policy.xlsx', engine="xlsxwriter")
    df.to_excel(writer, sheet_name='Simple_Taxi_Policy')
    writer.close()

    pretty_print_policy(env,my_policy)


if __name__ == '__main__':

    #'''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last', args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
    export_policy_to_excel(args.fpath, args.itr if args.itr >=0 else 'last', args.deterministic)
    #'''
    '''
    fpath = '/home/ari11/spinningup/ari_test/taxi_torch_basic_v1'
    env, get_action = load_policy_and_env(fpath, 'last')
    export_policy_to_excel(fpath, 'last')
    '''
