import torch
import random
import numpy as np
from collections import deque
import os
import json
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, BellmanTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 # use 0.001

class Agent:
    def __init__(self, load_checkpoint=True, checkpoint_file='bellman_checkpoint.pth'):
        self.n_games = 0
        self.gamma = 0.9 # use 0.9 
        self.epsilon = 1.0 # initial epsilon
        self.epsilon_max = 1.0 #use 1.0
        self.epsilon_min = 0.01 # use 0.01
        self.epsilon_decay = 0.005 # use 0.005
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = BellmanTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

        if load_checkpoint:
            self.load_checkpoint(checkpoint_file) # ini

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

    def save_checkpoint(self, filename='bellman_checkpoint.pth'):
        folder = './model'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filename='bellman_checkpoint.pth'):
        filepath = os.path.join('./model', filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint.get('n_games', 0)
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.memory = deque(checkpoint.get('memory', []), maxlen=MAX_MEMORY)
            print(f"Checkpoint loaded from {filepath}")
            print(f"Resumed from episode: {self.n_games}, epsilon: {self.epsilon}")
        else:
            print(f"No checkpoint found at {filepath}")


def train(load=True):
    agent = Agent(load_checkpoint=load)
    game = SnakeGameAI()
    plot_scores, plot_mean_scores = [], []
    total_score = 0
    record = 0
    episode_data = []

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

                if score > record:
                    record = score
                    agent.save_checkpoint('bellman_best.pth')

                # agent.save_checkpoint()

                total_score += score
                mean_score = total_score / agent.n_games
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

                # Simpan data episode
                episode_data.append({
                    'episode': agent.n_games,
                    'score': score,
                    'mean_score': mean_score,
                    'epsilon': agent.epsilon
                })

                # Simpan berkala
                if agent.n_games % 100 == 0:
                    with open(f'data/bellman_episode_data.json', 'w') as data:
                        json.dump(episode_data, data, indent=2)                    
                        agent.save_checkpoint(f'bellman_checkpoint_{agent.n_games}.pth')
                    

                print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Mean: {mean_score:.2f}, Epsilon: {agent.epsilon:.4f}')
                
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        agent.save_checkpoint('bellman_checkpoint.pth')
        print("Progress saved!")


if __name__ == '__main__':
    # Cek argumen untuk mode play atau continue
    start = input("Mulai training baru? (y/n): (n)").strip().lower()
    if start == 'y':
        print("Memulai training baru...")
        train(False)
    else:
        print("Melanjutkan training sebelumnya...")
        train(True)
