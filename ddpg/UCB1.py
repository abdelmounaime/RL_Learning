

class UCB1:
    
    def __init__(self, algos, epsilon):
        self.n = 0
        self.epsilon = epsilon
        self.algos = algos
        self.nk = np.zeros(len(algos))
        self.xk = np.zeros(len(algos))

    def select_best_algo(self):
        #check if nk == 0 or if we are in the first iterations
        for i in range(len(self.algos)):
            if self.nk[i] < 5:
                return np.random.randint(len(self.algos))

        return np.argmax([ self.xk[i] + np.sqrt(self.epsilon * np.log(self.n)/self.nk[i]) for i in len(algos)])

    def update_UCB1(self, algo_index, rew):
        self.xk[algo_index] = (self.nk[algo_index]*self.xk[algo_index] + rew)/(self.nk[algo_index]+1)
        self.nk[algo_index] += 1
        self.n += 1
