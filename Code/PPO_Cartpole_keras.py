import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import pylab
import numpy as np
import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.optimizers import Adam, RMSprop


def RL_Model(input_shape, action_space, lr):
    #in keras, every loss function takes arguments y_true(correct) and y_pred(model predicted)
    def ppo_loss(y_true, y_pred):#algorithm follows stadnart PPO clipped surrogate function structure
        advantages, prediction_picks, actions =y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:]
        LOSS_CLIPPING = 0.1
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = tensorflow.keras.backend.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss = -tensorflow.keras.backend.mean(tensorflow.keras.backend.minimum(p1, p2) + ENTROPY_LOSS * -(prob * tensorflow.keras.backend.log(prob + 1e-10)))
        return loss

    X_input = Input(input_shape)

    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)

    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss=ppo_loss, optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
    Actor.summary()
    Critic.summary()
    return Actor, Critic

    


class PPOAgent:
    #Pretty much the same as A2C, with the addition we need to create a structure to pass into the loss function
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.state_size=self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES=500
        self.env._max_episode_steps=400
        self.max=0#max for saving the best models 
        self.lr = 0.001

        self.states, self.actions, self.rewards,self.predictions = [], [], [], []
        self.scores, self.episodes, self.average = [], [], []
    
        # Create Actor-Critic network model with PPO loss function
        self.Actor, self.Critic = RL_Model(input_shape=(self.state_size,), action_space = self.action_size, lr=self.lr)


    def remember(self, state, action, reward,prediction):
        # store episode actions to memory
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)
        self.predictions.append(prediction)


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        #print(prediction)
        action = np.random.choice(self.action_size, p=prediction)
        return action, prediction

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
        predictions=np.vstack(self.predictions)
        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[0]
        # Compute advantages
        advantages = np.vstack(discounted_r - values)
        # training Actor and Critic networks
        #print(predictions)
        
        #print(np.shape(advantages))
        #print(np.shape(predictions))
        #print(np.shape(actions))
        #predictions=np.reshape(predictions,[1,-1])
        #y_true=Struct()
        #y_true.actions,y_true.predictions,y_true.advantage=actions,predictions,advantages
        #y_true=np.array(y_true)
        y_true=np.hstack([advantages,predictions,actions]) #we need to make them into an "array" so that we can access them simply by indexing
        #passing a Structure into loss function is unrecognized argument...bummer

        self.Actor.fit(states, y_true, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # env.reset training memory
        self.states, self.actions, self.rewards,self.predictions= [], [], [],[]#reset training memory
    
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save("cartpole-ppo.h5")
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
        
        pylab.savefig("PPO_Cartpole_v1.png")

        return self.average[-1]

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            #print(state)
            state = np.reshape(state, [1, self.state_size])
            #print(state)
            done, score= False, 0
            i=0
            while not done:
                #self.env.render()
                # Actor picks an action
                action,prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action)
                next_state=np.reshape(next_state, [1, self.state_size])
                # Memorize (state, action, reward) for training
                #if not done or i == self.env._max_episode_steps-1:
                    #reward = reward
                #else:
                    #reward = -100
                #prediction=np.reshape(prediction,[1,2])
                #action=np.reshape(action,[1,1])
                #print(np.shape(prediction))
                #print(np.shape(action))
                self.remember(state, action, reward,prediction)
                # Update current state
                state = next_state
                score += reward
                i+=1
                
                if done:
                    average = self.PlotModel(i, e)
                    if (i>=self.max):
                        self.max=i
                        self.save()
                        print("Saved")
                    # saving best models
                    #if i>=self.env._max_episode_steps:
                    #    self.save()
                    print("episode: {}/{}, score: {}, average: {:.2f}".format(e, self.EPISODES, score, average))

                    self.replay()
        # close environemnt when finish training
        self.env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(10):
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
    agent = PPOAgent(env_name)
    agent.run()
    #agent.test('cartpole-ppo.h5', '')