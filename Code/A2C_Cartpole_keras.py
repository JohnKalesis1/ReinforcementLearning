import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import pylab
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam, RMSprop

def RL_Model(input_shape, action_space, lr):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)

    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    #one model for predicting action and another for evaluating action
    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
    Actor.summary()
    Critic.summary()
    return Actor, Critic

class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.state_size=self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES=500
        self.env._max_episode_steps = 400
        self.lr = 0.0001
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
    
        # Create Actor-Critic network model
        self.Actor, self.Critic = RL_Model(input_shape=(self.state_size,), action_space = self.action_size, lr=self.lr)


    def remember(self, state, action, reward):
        # store episode actions to memory
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

                
    def replay(self):
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # env.reset training memory
        self.states, self.actions, self.rewards = [], [], []
    
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save("cartpole-a2c.h5")
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        
        pylab.savefig("A2C_Cartpole-v1.png")

        return self.average[-1]

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done, score= False, 0
            i=0
            while not done:
                #self.env.render()
                # Actor picks an action
                action = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action)
                next_state=np.reshape(next_state, [1, self.state_size])
                # Memorize (state, action, reward) for training
                #if not done or i == self.env._max_episode_steps-1:
                #    reward = reward
                #else:
                #    reward = -100
                self.remember(state, action, reward)
                # Update current state
                state = next_state
                score += reward
                i+=1
                
                if done:
                    average = self.PlotModel(i, e)
                    # saving best models
                    if i>=self.env._max_episode_steps:
                        self.save()
                    print("episode: {}/{}, score: {}, average: {:.2f}".format(e, self.EPISODES, score, average))

                    self.replay()
        # close environemnt when finish training
        self.env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                next_state, reward, done, _ = self.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = A2CAgent(env_name)
    agent.run()
    #models did not achieve max performanc, unlike DQN and DDQN, and thus we have no model saved
    #agent.test('cartpole-a2c.h5', '') 