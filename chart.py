import json
import matplotlib.pyplot as plt

# Load episode data
with open('episode_data.json', 'r') as f:
    data = json.load(f)

episodes = [d['episode'] for d in data]
scores = [d['score'] for d in data]
mean_scores = [d['mean_score'] for d in data]
epsilons = [d['epsilon'] for d in data]

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title('Score per Episode')
plt.plot(episodes, scores, label='Score', alpha=0.6, marker='o', markersize=2)
plt.plot(episodes, mean_scores, label='Mean Score', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title('Epsilon Decay')
plt.plot(episodes, epsilons, label='Epsilon', color='green')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
