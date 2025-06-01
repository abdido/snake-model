import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, BellmanTrainer
from helper import plot
import time
import sys

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self, load_model=False):
        self.n_games = 0
        self.epsilon = 0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = BellmanTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)
        
        # Load model if specified
        if load_model:
            self.load_model()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * self.n_games)

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, filename='bellmanBestModel.pth'):
        """Save model using trainer's save_model method"""
        return self.trainer.save_model(filename)

    def load_model(self, filename='bellmanBestModel.pth'):
        """Load model using trainer's load_model method"""
        return self.trainer.load_model(filename)

def train(load_previous=False, save_interval=100):
    plot_scores = []
    plot_mean_scores = []
    cumulative_scores = 0
    record = 0
    record_on = 0
    
    # Initialize agent with option to load previous model
    agent = Agent(load_model=load_previous)
    game = SnakeGameAI()
    start_time = time.time()
    longest_time = 0
    time_on = 0

    print('Training started...')
    if load_previous:
        print('Loading previous model...')
        agent.load_model('bellmanModel.pth')
        print(f'Continuing from episode {agent.n_games}')
    else:
        print('Starting new training session...')
        print('Press Ctrl+C to stop training and save progress.')
        print('Training will save checkpoints every', save_interval, 'episodes.')
        print('Use "python agent.py play" to play with the trained model.')



    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                cumulative_scores += score
                running_time = time.time() - start_time

                if score > record:
                    record = score
                    agent.save_model('bellmanBestModel.pth')  # Use agent's save_model method
                    record_on = agent.n_games

                # Save checkpoint every N episodes using trainer's save_model
                if agent.n_games % save_interval == 0:
                    agent.trainer.save_model(f'checkpoint_episode_{agent.n_games}.pth')

                if running_time > longest_time:
                    longest_time = running_time
                    time_on = agent.n_games

                plot_scores.append(score)
                mean_score = cumulative_scores / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

                print(
                    'Bellman Learning',
                    '\nEpisode:', agent.n_games,
                    '\nScore', score,
                    '\nHigh Score:', record, 'pada episode ke-', record_on,
                    '\nAll Scores:', cumulative_scores,
                    '\nMean Score:', mean_score,
                    '\nEpsilon:', agent.epsilon,
                    '\nTime:', running_time,
                    '\nLongest Time:', longest_time, 'pada episode ke-', time_on,
                    '\n')

                running_time = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        agent.trainer.save_model('bellmanModel.pth')
        print("Progress saved!")


if __name__ == '__main__':
    # Pilihan untuk training atau bermain
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        train(load_previous=False)
    elif len(sys.argv) > 1 and sys.argv[1] == 'continue':
        train(load_previous=True)
    else:
        train()  # Training baru