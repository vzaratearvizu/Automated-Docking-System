    episode/reward: Should have a general upward trend.

    train/mean_q1 and train/mean_q2: Should be steadily increasing together and with rewards. If rewards declines or plataeus while the mean_q1 keeps increasing, could be overestimate bias.

<br>

    train/alpha: A slow, sustained decline that doesn't crash to near-zero. It may settle at some moderate-non-collapsing value. It collapses when it suddenly drops rather than making a smooth decline into a plataeu.
    
    train/alpha_loss: Alpha loss often is the same shape as the line of Alpha, but not guaranteed.

<br>

    train/critic1_loss and train/critic2_loss: Should be trending down over time and stabilize but not hit exactly 0. Will stay stuck in as an amplitude bouncing around for awhile. See what happens after a positive reward ending episode occurs.

<br>

    Current Hypothesis - 

    train/mean_q: Parabola shape could be the critic learning to not overstimate anymore and produce values closer to what is really being rewarded.

    critic_loss: The sharp drop around 40k steps is due to the critics weights starting off as completely random. Critic just starts off untrained.