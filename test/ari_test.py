
from spinup import ppo_pytorch as ppo
#from spinup import ppo_tf1 as ppo
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#tf.config.set_visible_devices([], 'GPU')
import gym

#env_fn = lambda : gym.make('LunarLander-v2')

env_fn = lambda : gym.make('Taxi-v3')

ac_kwargs = dict(hidden_sizes=[64,64]) #, activation=tf.nn.relu)

logger_kwargs = dict(output_dir='ari_test/taxi_torch_6', exp_name='taxi_torch_6') #, cpu='auto')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=200, epochs=50, target_kl=0.1, logger_kwargs=logger_kwargs)