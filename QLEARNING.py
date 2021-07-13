import numpy as np
import gym
import matplotlib.pyplot as plt

def greedy(Q,s):
    return np.argmax(Q[s])

def eps_greedy(Q, s,eps=0.1):
    if np.random.uniform(0,1) < eps:
        return np.random.randint(Q.shape[1])
    else:
        return greedy(Q,s)

def run_episodes(env, Q, num_episodes=100, to_print=False):
    tot_rew = []
    state = env.reset()
    for _ in range(num_episodes):
        done = False
        game_rew = 0
        while not done:
            next_state, rew, done,_ = env.step(greedy(Q, state))
            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))
    else:
        return np.mean(tot_rew)

def one_episode_run(env, Q):
    state = env.reset()
    env.render()
    reward = 0
    done = False
    while not done:
        next_state, rew, done, _ = env.step(greedy(Q, state))
        state = next_state
        reward += rew
        env.render()

    print("Game reward : ", reward)

def Qlearning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma= 0.95, eps_decay=0.0005):
    nA= env.action_space.n
    nS= env.observation_space.n
    test_rewards = []
    Q = np.zeros((nS, nA))
    games_reward = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        if eps > 0.01:
            eps -= eps_decay

        while not done:
            action = eps_greedy(Q, state, eps)
            next_state, rew, done, _ = env.step(action)

            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state])-Q[state][action])
            state = next_state
            tot_rew += rew
            
            if done:
                games_reward.append(tot_rew)

        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 10000)
            print("Episode:{:5d} Eps:{:2.4f} Rew:{:2.4f}".format(ep, eps,test_rew))
            test_rewards.append(test_rew)
    
    return Q, test_rewards

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    env.reset()
    Q, test_reward = Qlearning(env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)
    
    x = np.arange(0,5000,300)
 
    plt.title("Reward evolution over episodes")
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.plot(x, test_reward, color="blue")
    plt.show()
    one_episode_run(env, Q)
