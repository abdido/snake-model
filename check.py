import torch

# Ganti path ini dengan path ke file Anda
file_path = "model\checkpoint_episode_5000.pth"

# Load file
checkpoint = torch.load(file_path, map_location="cpu")

# Tampilkan semua key yang disimpan
print("Isi dalam checkpoint:")
for key in checkpoint.keys():
    print("-", key)

# Jika reward log ada
if 'reward_log' in checkpoint:
    rewards = checkpoint['reward_log']
    print("\nContoh data reward:")
    print(rewards[:10])  # Cetak 10 nilai awal

    # Simpan ke file CSV (opsional)
    import pandas as pd
    df = pd.DataFrame({'Episode': list(range(1, len(rewards)+1)), 'Reward': rewards})
    df.to_csv("reward_log.csv", index=False)
    print("\nLog reward telah disimpan sebagai reward_log.csv")
else:
    print("\nTidak ditemukan key 'reward_log' dalam file.")
