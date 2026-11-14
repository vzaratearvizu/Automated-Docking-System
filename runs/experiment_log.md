# Experiment Log

Tracking variable changes that are made

## Change: Default. No Changes

Report: Critic_loss is bouncing between 1.5 and 4.5 with no signs of learning anything. The rewards are also staying in the range -4,000 to -3,000. It may be due to receiving no incentive to actually go towards the buoy, or that incentive is too small. It only learns this once it get's to the buoy but then that chance is completely random.

    Change File: rl_env.py
    Difference: 
    - Variable "DISTANCE_PENALTY_SCALE = 15" added
    - Variable "reward = -distance_ratio + reward_multiplier * 25" becomes
    "reward = -distance_ratio * DISTANCE_PENALTY_SCALE + reward_multiplier * 25"

Report: Reward was bumped down to around -57,000 with no signs of improving. Critics are largely disagreeing with one increasing to 40 with the other decreasing to 10. mean_q is stil decreasing. The distance penalty scale was too aggressive. Trying a square root curve instead so the penalty eases off faster as the distance shrinks rather than linearly.

    Change File: rl_env.py
    Difference: 
    - Variable "DISTANCE_PENALTY_SCALE = 15" removed
    - Variable "reward = -distance_ratio + reward_multiplier * 25" becomes
    "reward = -(distance_ratio ** 0.5) + reward_multiplier * 25"

Report: Square rooting was the wrong math thing to do. I should have squared it instead. Critic loss is unstable between 1.5 and 4.5, bouncing between them and the mean_q keeps decreasing while the rewards continue to be random.

    Change File: rl_env.py
    - Variable "reward = -(distance_ratio ** 0.5) + reward_multiplier * 25" becomes
    "reward = -(distance_ratio ** 2) + reward_multiplier * 25"

Report: A bit more successful. It hit positive values 6 times whereas last time it was less than 1. Could be due to luck, but I'll it seems to also fit that it worked when the alpha was high, possibly supporting the idea that if it stays more exploratory for longer then it'd reach the goal more.

    Change File: sac_neural_network.py
    - Variable "ALPHA_LEARNING_RATE = 0.00005" becomes "ALPHA_LEARNING_RATE = 0.00001"

Report: Viewed the checkpoints 250k through 750k and saw the agents actions were too random for how slow it takes for the boat to response. Increasing the response time for the movement of the boat and decreasing the range of options agent to prevent immediate switching from +1 to -1 and vice versa.
    
    Change File: boats.py
    - Variable: "THROTTLE_RESPONSE = 0.0002" becomes "THROTTLE_RESPONSE = 0.002"
    
    Change File: sac_neural_network.py
    - Variable: "LOG_STD_MAX = 2" becomes "LOG_STD_MAX = 0"

Report: The agent's actions are still bumping immediately from -1 to +1, so enforcing a harsher than before maximum standard deviation.

    Change File: sac_neural_network.py
    - Variable: "LOG_STD_MAX = 0" becomes "LOG_STD_MAX = -2.1"

Report: No improvement, but I did see the log maximum help out by preventing the throttle jumping from +1 to -1 and vice versa. Will also be bumping the log deviation minimum up and maximum up because we can only ever see about 2 decimal values being changed with the GUI and to slightly expand the deviation. It seems to not be learning very well, too slowly perhaps. Will be bumping down the batch size and it's barely used and seeing to many numbers at once.

    Change File: train.py
    - Variable: "BATCH_SIZE = 10000" becomes "BATCH_SIZE = 200"

    Change File: sac_neural_network.py
    - Variable: "LOG_STD_MIN = -20" becomes "LOG_STD_MIN = -2.3"
    - Variable: "LOG_STD_MAX = -2.1" becomes "LOG_STD_MAX = -0.8"

Report: No noticible improvements. mean_q is varying much more, the variation of the rewards is less. Next attempt will normalize the numbers. Normalization is important because without it, the same weights trying to learn the importance of each input are also being forced to learn the raw scale differences between them. So the weights are at the same time trying to figure out how to better fit 500 distance versus 0.12 turning velocity.

    Change File: environments.py
    - get_observation() now divides each value by its own real maximum
      (pi for the angles, MAX_SPEED, MAX_TURN_RATE, and a MAX_POSSIBLE_DISTANCE
      constant) before returning the array, putting every value into roughly
      the same -1 to 1 range.

Report: Rewards stayed in the same general area, improved for a sustained
stretch (14+ genuine docks confirmed in the terminal log, the first
repeated successes seen across this whole project), then had a steep
decline along with increasing variance, mirrored in mean_q. Traced this to
a real bug: distance_ratio = current_distance / starting_distance can
blow up whenever a random spawn happens to place the boat very close to
the buoy (tiny starting_distance), the exact same physical drift then
produces a wildly inflated ratio purely from dividing by a small number,
not because the boat actually behaved worse. This explains the extreme
outlier timeout values found in the log (-21,952, -53,386, -28,594, etc.),
several times larger than a normal timeout.

Separately, mean_q has also been declining across many prior changes while
reward stays flat/noisy which is is still an open, SEPARATE hypothesis (recursive
bootstrapping with no reliable terminal grounding, not yet confirmed by
this particular finding) to test later via GAMMA, once this distance_ratio
bug is fixed and isolated on its own.
    
    Change File: rl_env.py
    - Variable: "distance_ratio = current_distance / self.starting_distance" becomes "distance_ratio = min(current_distance / self.starting_distance, 2.0)"

Report: Up to around episode 241 and 853,000 steps the reward climbs before collapsing and the q_mean slowly decreases here before collapsing which happens to occur down to alpha 0.57. Figure out some way to replicate this but for longer in next experiment. For now, let's see if a gamma change does anything for the q_mean

    Change File: sac_neural_network.py
    - Variable: "GAMMA = 0.99" becomes "GAMMA = 0.95"

Report: Will be increasing the steps by 2x, 2 million to 4 million. It seems the q_mean finally learns where it's suppose to go around 1.5 million so this may help the model learn more. Will also be choosing a value between 0.99 and 0.95 since at 0.95 it begins to go down and at 0.99 it goes up around the 1.5 million mark in q_mean.

    Change File: sac_neural_network.py
    - Variable: "GAMMA = 0.95" becomes "GAMMA = 0.96"

    Change Files: train.py
    - Variable: "TOTAL_STEPS = 2000000" becomes "TOTAL_STEPS = 4000000"

Report: No improvement except learning the gamma affects q_mean. mean_q dipped (~-42 at 600k)
and recovered (~-20 by 1M) without reward following. Creating an entire new baseline to see how the model performs. Also added new trackers for how many boats have docked in total across training and per-episode docked flag (0/1) whose TensorBoard-smoothed line works
as a rolling success rate.

    Change File: sac_neural_network.py
    - Variable: "ACTOR_LEARNING_RATE = 0.00001" becomes "ACTOR_LEARNING_RATE = 0.0003"
    - Variable: "CRITIC_LEARNING_RATE = 0.00001" becomes "CRITIC_LEARNING_RATE = 0.0003"
    - Variable: "ALPHA_LEARNING_RATE = 0.00001" becomes "ALPHA_LEARNING_RATE = 0.0003"
    - Variable: "GAMMA = 0.96" becomes "GAMMA = 0.99"
    - "target_entropy = -5" becomes "target_entropy = -2"

    Change File: train.py
    - Variable: "BATCH_SIZE = 200" becomes "BATCH_SIZE = 256"

Report: Selecting models by best episode reward is unreliable — checkpoints saved at
higher episode rewards performed worse (likely lucky close spawns). Learned to dock early (best early
checkpoint 26/50 deterministic), then collapsed — rolling docked rate ~0.54 mid-run
→ ~0.009 at end. mean_q rose well above 0 (first time), alpha_loss ~0 (equilibrium),
alpha rising from ~0.25. Critic loss exploded around 900k (~3,400), but eval shows
policy already degrading by 500k (15/50). Climb-then-collapse persists;
critic divergence is the likely cause, reward scale (zone_progress*25) the likely
trigger. Will add a reward scale to decrease the extremity of sudden rewards from docking. Smaller values make it easier for the critics to predict accurately, since errors become smaller and gradient updates gentler. This keeps overestimation errors from compounding through the critics’ own targets and inflating Q-values, as happened in the last run. Reviewing checkpoints during training by hand in separete terminal to avoid halt in training and view potential better solutions that happen earlier. We're also going to undo the reward scaling in the logging for TensorBoard to compare it to previous runs.

    Change File: train.py
    - Variable: "CHECKPOINT_EVERY = 500000" becomes CHECKPOINT_EVERY = 250000
    - Variable: "episode_reward += reward" becomes "episode_reward += reward / REWARD_SCALE"

    Change File: rl_env.py
    - Addition: "REWARD_SCALE = 0.1"
    - Line: "return observation, reward, terminated, truncated, info" becomes "return observation, reward * REWARD_SCALE, terminated, truncated, info"

Report: Saving the model's checkpoints by how many points it score is not all good. It saves only its luckiest moments when it spawns closest to the buoy. A complete redesign of the environment, rewards, and how models are saved are the next steps. Because of how big this change will be, the changes will be viewed through the github pushes rather than written here. 