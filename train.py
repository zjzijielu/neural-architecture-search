import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--l', type=int, default=4, help="num of layers")
parser.add_argument('--t', type=int, default=100, help="num of models generated")
parser.add_argument('--e', type=int, default=10, help="num of epoch to train")
parser.add_argument('--b', type=int, default=128, help='batch size')
parser.add_argument('--exp', type=float, default=0.8, help="exploration prob")
parser.add_argument('--reg', type=float, default=1e-3, help="regularization strength")
parser.add_argument('--cc', type=int, default=32, help="num of cells in RNN Controller")
parser.add_argument('--dim', type=int, default=20, help="embedding dimension")
parser.add_argument('--acc_beta', type=float, default=0.8, help="beta for exponential moving average")
parser.add_argument('--clip', type=float, default=0.0, help="if clip the rewards to [-0.05, 0.05]")
parser.add_argument('--r', type=bool, default=True, help="restore controller to continue training")
parser.add_argument('--g', type=float, default=0.5, help="weight for num params reward")
parser.add_argument('--rnp', type=int, default=1, help="include num params as reward")

args = parser.parse_args()


# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = args.l  # number of layers of the state space
MAX_TRIALS = args.t  # maximum number of models generated

MAX_EPOCHS = args.e  # maximum number of epochs to train
CHILD_BATCHSIZE = args.b  # batchsize of the child models
EXPLORATION = args.exp  # high exploration for the first 1000 steps
REGULARIZATION = args.reg  # regularization strength
CONTROLLER_CELLS = args.cc  # number of cells in RNN controller
EMBEDDING_DIM = args.dim  # dimension of the embeddings for each state
ACCURACY_BETA = args.acc_beta  # beta value for the moving average of the accuracy
CLIP_REWARDS = args.clip  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = args.r  # restore controller to continue training
GAMMA = args.g 
REWARD_NUM_PARAMS = args.rnp

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA, gamma=GAMMA, rnp=REWARD_NUM_PARAMS)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc, prev_num_p = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, prev_num_p, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)