# Reinforcement Learning Timelapses Reproducibility

## Bipedal Walker
```
python train_deep_rl.py --algo SAC --env BipedalWalker-v3 --seed 1 --start_timesteps 5000 --max_episode_steps 500 --eval_freq 10000 --max_timesteps 1000000 --policy_freq 2 --eval_episodes 1 --batch_size 256 --save_model --save_video
```

[![](https://markdown-videos-api.jorgenkh.no/youtube/CMij26OdtZ4)](https://youtu.be/CMij26OdtZ4)

## Lunar Lander
```
python train_deep_rl.py --algo SAC --env LunarLander-v3 --seed 1 --start_timesteps 5000 --max_episode_steps 1000 --eval_freq 5000 --max_timesteps 2000000 --policy_freq 2 --eval_episodes 1 --batch_size 512 --save_model --save_video

```
