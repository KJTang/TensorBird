import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import random

import sys
sys.path.append("game/")
# from flappybird import GameApp
from flappybird_test import GameApp     # use simplified game to test first
game = GameApp();
game.Init();

kActionFlap = [0, 1];
kActionStay = [1, 0];
kActionPool = [
    kActionStay, 
    kActionFlap
];
kActionCnt = len(kActionPool);

kGamma = 0.99           # decay rate of past observations
kObserve = 100000;      # timesteps before training
kExplore = 2000000;     # frames over to anneal epsilon
kEpsilonInit = 0.0001;
kEpsilonFinal = 0.0001;

kReplayMemSize = 50000;
kMiniBatchSize = 32;

kFramePerAction = 1;    # passed kFramePerAction frames, take 1 action

kSavePath = "flappybird";
kSaveInterval = 10000;

DEBUG = True;
if DEBUG: 
    kObserve = 2000;
    kExplore = 20000;
    kEpsilonInit = 0.9000;
    kEpsilonFinal = 0.0001;
    kSavePath = "flappybird_test";
    kSaveInterval = 1000;

def TickGame(action = kActionStay): 
    if DEBUG: 
        image_data, reward, terminal = game.ManualGameLoop(action); 

        # image
        image_data = cv2.cvtColor(cv2.resize(image_data, (4, 4)), cv2.COLOR_BGR2GRAY);
        ret, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY);
        image_data = np.reshape(image_data, (4, 4, 1));

        return image_data, reward, terminal;
        

    image_data, reward, terminal = game.ManualGameLoop(action); 

    # resize & convert image data
    image_data = cv2.cvtColor(cv2.resize(image_data, (64, 64)), cv2.COLOR_BGR2GRAY);
    ret, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY);
    image_data = np.reshape(image_data, (64, 64, 1))

    return image_data, reward, terminal;


def CreateNetwork(): 
    if DEBUG: 
        input_layer = tf.placeholder("float", [None, 4, 4, 1]);
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=4,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        pool1_flat = tf.reshape(pool1, [-1, 2 * 2 * 4])
        output_layer = tf.layers.dense(inputs=pool1_flat, units=kActionCnt);
        return input_layer, output_layer;

    input_layer = tf.placeholder("float", [None, 64, 64, 1]);

    # input: 64 * 64 * 1
    # output: 64 * 64 * 32
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 8],
        padding="same",
        activation=tf.nn.relu
    )
    # input: 64 * 64 * 32
    # output: 32 * 32 * 32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # input: 32 * 32 * 32
    # output: 32 * 32 * 64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu
    )
    # input: 32 * 32 * 64
    # output: 16 * 16 * 64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # input: 16 * 16 * 64
    # output: 16 * 16 * 64
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    # input: 16 * 16 * 64
    # output: 8 * 8 * 64
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])

    # input: 8 * 8 * 64
    # output: 1024
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
    # output_layer = tf.layers.dense(inputs=dropout, units=kActionCnt)

    # input: 1024
    # output: 2
    output_layer = tf.layers.dense(inputs=dense, units=kActionCnt)

    return input_layer, output_layer;


def TrainNetwork(input_layer, output_layer, sess): 
    # define the loss function
    a = tf.placeholder("float", [None, kActionCnt])
    y = tf.placeholder("float", [None])
    y_ = tf.reduce_sum(tf.multiply(output_layer, a), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    # store the previous observations in replay memory
    replayMem = deque()

    # init first state
    s_t, reward, terminal = TickGame();

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
        
    # start training
    epsilon = kEpsilonInit
    t = 0
    while True:
        # choose an action epsilon greedily
        output_t = output_layer.eval(feed_dict={input_layer : [s_t]})
        a_t = kActionStay
        action_index = 0
        if t % kFramePerAction == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(kActionCnt);
            else:
                action_index = np.argmax(output_t)
            a_t = kActionPool[action_index];

        # scale down epsilon
        if epsilon > kEpsilonFinal and t > kObserve:
            epsilon -= (kEpsilonInit - kEpsilonFinal) / kExplore
            epsilon = epsilon if epsilon > kEpsilonFinal else kEpsilonFinal;

        # run the selected action and observe next state and reward
        s_t_next, r_t, terminal = TickGame(a_t);

        # store the transition in replay memory
        replayMem.append((s_t, a_t, r_t, s_t_next, terminal))
        if len(replayMem) > kReplayMemSize:
            replayMem.popleft()

        # only train if done observing
        if t > kObserve:
            # sample a minibatch to train on
            minibatch = random.sample(replayMem, kMiniBatchSize)

            # get the batch variables
            s_batch     = [d[0] for d in minibatch]
            a_batch     = [d[1] for d in minibatch]
            r_batch     = [d[2] for d in minibatch]
            s_next_batch  = [d[3] for d in minibatch]

            y_batch = []
            a_next_batch = output_layer.eval(feed_dict = {input_layer : s_next_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + kGamma * np.max(a_next_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                input_layer : s_batch}
            )

        # update the old values
        s_t = s_t_next
        t += 1

        # save progress 
        if t % kSaveInterval == 0:
            saver.save(sess, 'saved_networks/' + kSavePath + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= kObserve:
            state = "observe"
        elif t > kObserve and t <= kObserve + kExplore:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, \
            "/ STATE", state, \
            "/ EPSILON", epsilon, \
            "/ ACTION", action_index, \
            "/ REWARD", r_t, \
            "/ OUTPUT", output_t)



def main(): 
    # game = GameApp();
    # game.Init();
    # game.AutoGameLoop();

    sess = tf.InteractiveSession()
    input_layer, output_layer = CreateNetwork()
    TrainNetwork(input_layer, output_layer, sess)

if __name__=="__main__":
    main() 