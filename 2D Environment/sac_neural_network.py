# Soft Actor-Critic Neural Network
# Basically, one neural network will make decisions and two others will criticize it.
# The lower score will be used to avoid trusting an overly optimistic single critic (stops overestimation bias).
# Overestimation could lead to the actor exploiting something the critic says is good.
# Actor's job is to find out what gives a higher score from the critic.
# The critic learns to predict a value, estimating future rewards from previous good values seen.

import copy
import numpy
import torch
import torch.nn
import torch.nn.functional
import torch.optim

# From get_observation() in environments.py
# heading_relative_angle, angle_to_target, boat.speed, boat.angular_velocity, distance
OBSERVATION_DIMENSION = 5
# Throttle and steering
ACTION_DIMENSION = 2
# A log_standard_deviation learned by the AI. Have a min and max to prevent
# absurdly large numbers that make it unstable.
# When deciding this, literally look at e^(the min or max number here)
LOG_STD_MIN = -20
LOG_STD_MAX = 2
# A percentage that will determine how much the target network's parameters are changed towards the critics current values
# A low value is preferred to help the critic learn without whiplashing from low to high values, it's 
# instead a small percentage moving towards the critics value
# TAU is just a greek letter variable τ
TAU = 0.005
# How quickly the learning will occur
ACTOR_LEARNING_RATE = 0.00001
CRITIC_LEARNING_RATE = 0.00001
# Learning rate for the alpha parameter. Controls how big adjustments for the parameter are.
# This rate never changes, just the alpha parameter changes.
ALPHA_LEARNING_RATE = 0.00005
# Gamma is the discount factor, how much it should car about rewards farther in the future.
# Values closer to 1.0 means care more deeply about the future reward. 0.0 means only care about the next reward
GAMMA = 0.99

class Actor(torch.nn.Module):
    def __init__(
        self, observation_dimension=OBSERVATION_DIMENSION, action_dimension=ACTION_DIMENSION):
        super().__init__()

        # nn.linear() does output = input * weights + bias
        # This tiny area is meant for figuring out what to do with the observations
        self.layer1 = torch.nn.Linear(observation_dimension, 380)
        self.layer2 = torch.nn.Linear(380, 285)
        self.layer3 = torch.nn.Linear(285, 143)
        self.layer4 = torch.nn.Linear(143, 36)
        self.relu = torch.nn.ReLU()

        # mean_layer is the center of distribution that the action will be sampled from
        # log_standard_deviation says how far away the value can move to from the center distribution
        # This tiny area is meant for figuring out what to do with the actions after seeing the observations
        self.mean_layer = torch.nn.Linear(36, action_dimension)
        self.log_standard_deviation_layer = torch.nn.Linear(36, action_dimension)

    # relu is done here rather than with a sequential function since we had to fork the neural network into the mean_layer and log_standard_deviation_layer
    # If we were to put it inside a sequential, then we'd have no way of producing more than one value like Sequential does
    def forward(self, observation):
        x = self.relu(self.layer1(observation))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        # The center of distribution means a sampeled number (randomy) will be where it most likely becomes the next action choice
        # for throttle and steering
        mean = self.mean_layer(x)
        # The deviation spreads how far this randomness will be for choosing a sample.
        # This will be the spread logged, so exponent(log_standard_deviation).
        # We do an exponent because the standard deviation always has to be a positie number because
        # there is no such thing as a negative spread. Belle currve iso either narrow or wide, not negatively inverted.
        # exp() means e raised to something, so e^(log_standard_deviation)
        log_stan_devi = self.log_standard_deviation_layer(x)
        return mean, log_stan_devi

    def sample(self, observation):
        # Calls forward with the observation to receive a mean and log_standard_deviation to make a decision with
        # Like creating a reference point/area where moves can be made and then feeding it into something else that
        # actually makes a decision (value between -1 and +1)
        mean, log_stan_devi = self.forward(observation)
        # Clamp is (input, min, max). Useful for providing a guardrail since the number could literally be anything from
        # the AI as it's trying to learn. Providing a range to what it the learned deviation for exploring is.
        log_stan_devi = torch.clamp(log_stan_devi, LOG_STD_MIN, LOG_STD_MAX)
        # Basically e^(log_standard_deviation)
        standard_deviation = log_stan_devi.exp()

        # Builds a bell-curve. The mean at the center then however far it can be from e^(log_stan_devi)
        # Creates a normal guassian (even spread)
        normal_distribution = torch.distributions.Normal(mean, standard_deviation)
        # Chooses a random number in the range of the curve
        # rsample over sample because we couldn't learn how changing the weights would affect the random
        # rsample uses mean + standard_deviation * noise. The noise is the random part whereas mean + standard_deviation is learned, so
        # we can see how changing the weights could potentially affect the outcoming action value.
        unbounded_action = normal_distribution.rsample()
        # Bounds the previous unbounded_action. Whatever the value is (x), produces a y (between -1 and +1) that is learned to input
        # as an action value
        bounded_actions = torch.tanh(unbounded_action)

        # So in the bell curve and since it was evenly distributed, it finds where the value y ended up and uses the x
        # axis to figure out what the probability of getting that value is
        log_probability = normal_distribution.log_prob(unbounded_action)
        # If bounded_actions happen to be +1 or -1, then log(1-1^2) gives 0 then log 0 is -infinity. No good, breaks.
        # We're doing log (a/b) = log(a) - log(b) with the .log_prob and the .log above and below.
        # density_of_squashed_value = density_of_raw_value / derivative | log(density_of_squashed) = log(density_of_raw) - log(derivative)
        # The density is how big the probability was of choosing that value. Imagine a bell-curve, the
        # density is that space between the y and x axis.
        # Derivitive of tanh(x) is 1 - tanh(x)^2.
        # The reason there is a number added afterwards within the log is because we need to prevent -infinity from happening
        # A small value is needed so it doesn't meaningfully distort result of the bounded_action as an actual value that believes
        # should be used in the learning. 1e-8 = 0.00000001, just small enough to avoid negative infinity.
        log_probability -= torch.log(1 - bounded_actions.pow(2) + 1e-8)
        # Addimg the steerng and throttle together. Adding together because they are independent of each, if they
        # relied on each other we'd do multiplication. Adding together to get an overarching view of just how random the
        # value we ended up with is.
        log_probability = log_probability.sum(1, keepdim=True)

        # This is just the decision without any randonmness math. Has nothing to do with the training, this value is used
        # when using the watch_trained.py to view the AI's general performance when doing performance checks.
        deterministic_action = torch.tanh(mean)

        # Returning sampled actions.
        # bounded_actions is th real, usable action for steering and throttle
        # log_probability is for the combined probability of how random was the sampled_actions
        # deterministic_action for watching the training checkpoints
        return bounded_actions, log_probability, deterministic_action


class Critic(torch.nn.Module):
    def __init__(self, observation_dimension=OBSERVATION_DIMENSION, action_dimension=ACTION_DIMENSION):
        super().__init__()
        self.net = torch.nn.Sequential(
            # Putting observation and action together to figure out if it was a good decision to make given the observation
            torch.nn.Linear(observation_dimension + action_dimension, 380),
            torch.nn.ReLU(),
            torch.nn.Linear(380, 285),
            torch.nn.ReLU(),
            torch.nn.Linear(285, 143),
            torch.nn.ReLU(),
            torch.nn.Linear(143, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 1),
        )

    # relus are inside the Sequential function since we aren't forking anything, it can all be done in there since we only need one value.
    def forward(self, observaation, action):
        # concatenates them together. Order doesn't matter, just keep consistent
        return self.net(torch.cat([observaation, action], dim=1))


# This is a one-directional communication, where the source_network gives its parameters to the target network
def soft_update(source_network, target_network, tau=TAU):
    # We can grab all the parameters (the weights and biases) of neural networks with .parameters() since they are stored by torch
    # Using zip(x,y) here make it so parameters from both networks are paired together, pairing each layer together
    for source_parameter, target_parameter in zip(
        source_network.parameters(), target_network.parameters()
    ):
        # .data access numbers without involving PyTorch's gradient-tracking.
        # .copy overwrites the target_parameters with the source_parameter's parameters
        # (percentage from source * source_parameters + (remaining amount to target networks parameters))
        # An amount of the previous source parameter + what's already in the target network to update the target network
        target_parameter.data.copy_(
            tau * source_parameter.data + (1 - tau) * target_parameter.data
        )


# Stores past expriences so training can randomly sample batches from it.
# An experience would be one step and there are however many ticks in a second, so however many experiences per second.
# Holds data in place until ready to use for training so it can be randomly sampled over and over again.
# Good for randomly sample because the neural network will not overfit on patterns, but will instead give random
# pieces of episdoes that may not correlate, which it will then learn what tends to work and not in general.
class ReplayBuffer:
    # Capacity is a the maximum number of past experiences the buffer can hold before overwriting the oldest ones
    def __init__(self, capacity, observation_dimension=OBSERVATION_DIMENSION, action_dimension=ACTION_DIMENSION):
        self.capacity = capacity
        # Where the next entry will be written
        self.pointer = 0
        # How many entries that have been filled in so far
        self.size = 0

        # Creates a table for observations, actions, rewards, what the world would be after action is taking, and
        # if the current step resulted in a finished episode (in success by docking)
        self.observations = numpy.zeros(
            (capacity, observation_dimension), dtype=numpy.float32
        )
        self.actions = numpy.zeros((capacity, action_dimension), dtype=numpy.float32)
        self.rewards = numpy.zeros((capacity, 1), dtype=numpy.float32)
        self.next_observation = numpy.zeros(
            (capacity, observation_dimension), dtype=numpy.float32
        )
        self.completed = numpy.zeros((capacity, 1), dtype=numpy.float32)

    # Adds things to the table every single step
    def add(self, observation, actions, reward, next_observation, completed):
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = reward
        self.next_observation[self.pointer] = next_observation
        self.completed[self.pointer] = completed

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # Randomly samples something from the buffee once there's enough to pull a batch from
    def sample(self, batch_size):
        indices = numpy.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observation[indices],
            self.completed[indices],
        )

    # Returns the length size for comparison to see if the buffer is large enough for a batch
    def __len__(self):
        return self.size

# One agent, consisting of 5 neural networks
class SACAgent:
    def __init__(self, observation_dimenison=OBSERVATION_DIMENSION, action_dimension=ACTION_DIMENSION,
        actor_learning_rate=ACTOR_LEARNING_RATE, critic_learning_rate=CRITIC_LEARNING_RATE,
        alpha_learning_rate=ALPHA_LEARNING_RATE,
        gamma=GAMMA, tau=TAU, device="cuda"):

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Just setting up the actor neural network
        self.actor = Actor(observation_dimenison, action_dimension).to(device)
        # The actual updating of weights and biases. Like Stoachastic Gradient Descent, for example.
        # Adam is a type of optimizer/weight updater.
        # It adds a momentum and adaptive per-parameter step size
        # m = moving average of gradient. What direction has gradient been generally moving
        # v = moving average of squared gradient. How noisy has the parameter's gradient magnitude generally been
        # t is the current time step
        # So for each parameter,
        # m = constant * m + (1 - constant) * gradient
        # m_corrected = m / (1 - constant^t)
        # v = constant * v + (1 - constant) * gradient^2
        # v_corrected = v / (1 - constant^t)
        # weight -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

        # We have two critics. Could add more critics and still choose the least optimistic value or an average
        # of all the critics.
        self.critic1 = Critic(observation_dimenison, action_dimension).to(device)
        self.critic2 = Critic(observation_dimenison, action_dimension).to(device)
        # The actual updating of weights and biases. Like Stoachastic Gradient Descent, for example.
        # Adam is a type of optimizer/weight updater.
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_learning_rate)

        # The target critics are there to provide a stable copy of the real critics.
        # The stable copy is useful for computing the gradients since we're providing a way to not get a
        # loop of upating the netwoork. The critic would compute a target value of what it thinks the
        # future would be. As the critic updates its weights, so do the critics target value which would
        # loop over and over again since they are both moving at the same time rather than producing a
        # target value and adjusting weights to get therre.
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # target_entropy = target randonmess that never changes and is only ever used in alpha_loss.
        # alpha_loss produces a graadient that adjusts alpha.
        # alpha is a scalar that controls how random/exploratory the actor should be and it's used in
        # alpha_optimizer alpha_loss, and soft_target.
        # action_dimension here is a convention, does not have to be action dimension.
        # The idea comes from the log_probability as it's producing a value of how likely it was to produce the sample
        # from the bell curve containing possible actions to make.
        # A log_probability producing negative is more spread out randomness whereas producing positive is more
        # tight-knit confident. So in a way we're just choosing the constant for randomness that will
        # be used for the duration of the training. Alpha is what chooses how much to take into consideration
        # the target_entropy and is adjusted throughout trraining.
        self.target_entropy = -5
        # Requiring a gradient here since we'll be updating weights with log_alpha.
        # Also creating a table of zeros that's 1x1 because there's only one alpha being used for
        # the two actions (steering and throttle) together. Make the table bigger if wishing for an alpha
        # for particular actions. Alpha starts at 0, meaning exponent 0 is 1, so a neutral scalar that
        # weights the critic-score and entropy randomness the same way. Choosing between what the critic says
        # should be a good value to chase versus choosing a random good value for exploring.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # Adam is a type of optimizer/weight updater.
        # Creates an optimizer Adam object and we're telling it it is respoonsible for log_alpha and that
        # the adjustments should be made with the alpha_learning_rate
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)

    # We use @property for just convenience and labeling, otherwise it'd be e^log_alpha instead of a labeled
    # "oh, this is the alpha"
    # @property also helps with rerunning it rather than having before it's used alpha = e^log_alpha cause
    # then we'd have multiple for each time it's used.
    @property
    def alpha(self):
        # exp() is e^log_alpha
        return self.log_alpha.exp()

    # Uses the actor neural network to make a decision
    # The evaluate argument is false by default for training and is set to true when viewing progress
    # checkpoints for the neural network.
    def select_action(self, observation, evaluate=False):
        # FloatTensor just converts a normal numpy array into a PyTorch array. PyTorch array's only work within
        # the Acttor network. PyTorch math is needed here such as the linear function and ReLU with the neurons.
        # Unsqueeze takes the existing 7 numbers in the observation and adds another layer to turn it into a matrix.
        # These numbers change from [x_pos, y_pos, heading, angle_to_target, speed, angular_velocity, distance] which is shape (7,)
        # to shape [[x_pos, y_pos, heading, angle_to_target, speed, angular_velocity, distance]] which is shape (1,7).
        # Just an extra pair of brackets that changes its dimension. The way the neural network works with the tensor observaiton
        # is that it register (1,7) as "There is only one observation, and this one observation has 7 numbers."
        # If it were (2,7), it'd mean it's processing two observations at once with 7 numbers in each. It could would
        # with zeros as padding if there were less than 7 numbers for some other observation component.
        # Makes it to the current device because that's where it's holding the values, otherwise numbers woud be
        # separated between the CPU and GPU and would need additional operaions to communicate between the two.
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        # Don't go through the gradient because we're only selecting a random choice, not learning from it yet.
        with torch.no_grad():
            # Calling sample from the actor returns a tuple of 3. We just need action and determinsitic_action
            # and not log_probability as this select_action function is only selecting an action.
            # We'll use action for training and deterministic_action for when viewing it back as a progress check.
            # The action one had extra stuff happening like randomly sampling from the bell curve.
            action, log_probability, deterministic_action = self.actor.sample(observation)
        # Choose the random action by default unless going through the progress check.
        chosen_action = deterministic_action if evaluate else action
        # Need to choose the cpu here because numpy doesn't know how to work with GPUs, but CPUs is a yes.
        # The chosen_action is a tensor array made from the network. And we only need what's at index 0
        # because it removes the fake batch dimension from unsqueeze.
        return chosen_action.cpu().numpy()[0]

    # This updates the weights of the neural networks
    # The batch_size controls how many renadom experiences get pulled from the replay buffer and work together to update the weights.
    # Conventially 256 but can be anything that works best for the network.
    # Whatever batch_size is, is how many random experiences get it pulled from the replay_buffer
    def update(self, replay_buffer, batch_size):
        # These states are used in the replay buffer and grabs however many random experiences
        observations, actions, rewards, next_observations, completed = replay_buffer.sample(batch_size)

        # Sends all to the selected device or else it'd have to be cross-communicating which is extra steps.
        # Converted to a Tensor so the neural networks can work with them
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        completed = torch.FloatTensor(completed).to(self.device)

        # We are not updating anything here. We're simply calculating what the next action and target values will be
        # for the actual gradient update since we need them there.
        with torch.no_grad():
            # deterministic_action not needed. Just good to see what's suppose to be there instead of elaving empty
            # Needed the next action and probability of choosing it for the soft target
            next_actions, next_log_probs, deterministic_action = self.actor.sample(next_observations)

            # Uses the two target neural networks of the critics so I don't create a moving goal post. Helps keep a solid target value
            target_q1 = self.target_critic1(next_observations, next_actions)
            target_q2 = self.target_critic2(next_observations, next_actions)
            # Choose the least optimistic choice. A least optimistic choice produce a least optimistic reward prediction
            target_q_min = torch.min(target_q1, target_q2)

            # Soft refers to softening the target_q_min, the reward chosen. A non-soft or hard target would just be target_q_min.
            # Its softened by minusing with entropy. This entropy has nothing to do with the target_entropy. This entropy is specific
            # to making one from the neural network we don't want to be a moving goal post. The alpha is a weight being learned which
            # is choosing between what the critic says should be a good value to chase versus choosing a random good value for exploring,
            # multiplied by the probability of choosing the action from the next_observations.
            # So hard target is being subtracted by the probability of taking an action multiplied by how much we want to explore or exploit
            soft_target = target_q_min - self.alpha * next_log_probs
            # The complete Bellman equation. Bellman equation is just a recursive formula that shows immediate rewards and future rewards.
            # Instant reward + 0 if we completed the game with these observations *  how much we care about future rewards *
            # what is expected of the next_observation by finding the probability of making a decision from exploriong or exploiting.
            target_value = rewards + (1 - completed) * self.gamma * soft_target

        # Enters the current batch observations and batch actions to critic all of them
        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)

        # For each batch, (current_q - target_value)^2 to cancel positive and negative values, as well as disproportionately penalize larger
        # errors. This one is for just adjusting to getting to the target value, like adjusting a slope of a line to multiple data points.
        critic1_loss = torch.nn.functional.mse_loss(current_q1, target_value)
        critic2_loss = torch.nn.functional.mse_loss(current_q2, target_value)

        # This is to get rid of previous gradients that were calculated. PyTorch doesn't automatically clear them, so without this,
        # it'd result in gradients stacking (adding) on top of each other
        self.critic1_optimizer.zero_grad()
        # Backpropogatting to find the slope of each weight and forms a bias, the gradient.
        critic1_loss.backward()
        # After calculating the gradients, PyTorch doesn't automatically begin reading and updating the weights, so this is like
        # running a stream onto the rocky gradients to then reach all the weights. 
        # It takes the gradients from backward and runs them through Adam's update formula with momentum, average momentum, 
        # adaptive-step-size math.
        self.critic1_optimizer.step()

        # This is to get rid of previous gradients that were calculated. PyTorch doesn't automatically clear them, so without this,
        # it'd result in gradients stacking (adding) on top of each other
        self.critic2_optimizer.zero_grad()
        # Backpropogatting to find the slope of each weight and forms a bias, the gradient.
        critic2_loss.backward()
        # After calculating the gradients, PyTorch doesn't automatically begin reading and updating the weights, so this is like
        # running a stream onto the rocky gradients to then reach all the weights. 
        # It takes the gradients from backward and runs them through Adam's update formula with momentum, average momentum, 
        # adaptive-step-size math.
        self.critic2_optimizer.step()

        # deterministic_action is not needed since we're updating weights, not viewing through a checkpoint. Nice to see there.
        # This actor's gives us the actions and the probability of choosing those actions.
        new_actions, log_probabilitys, deterministic_action = self.actor.sample(observations)
        # We feed into the critic to judge how well our actions were given the observations
        q1 = self.critic1(observations, new_actions)
        q2 = self.critic2(observations, new_actions)
        # Choose the least optimistic one
        q_min = torch.min(q1, q2)

        # detach() make a copy of the alpha, multiplies it by the probability of making the decisions minus the least optimistic score, 
        # then take the average of all of them. We make a copy of alpha because we don't want any of the gradients with it when
        # calculating the loss. By multiplying how much we want to adjust the balance between exploring and exploiting to the
        # probability of making those decisions, we then subtract the current real critics' score of what it thinks of the current 
        # actions being made because we want to minimize the loss, so subtracting a large score gives a smaller loss. We then
        # average them all out to see how in general the model is performing.
        actor_loss = (self.alpha.detach() * log_probabilitys - q_min).mean()

        # This is to get rid of previous gradients that were calculated. PyTorch doesn't automatically clear them, so without this,
        # it'd result in gradients stacking (adding) on top of each other
        self.actor_optimizer.zero_grad()
        # Backpropogatting to find the slope of each weight and forms a bias, the gradient.
        actor_loss.backward()
        # After calculating the gradients, PyTorch doesn't automatically begin reading and updating the weights, so this is like
        # running a stream onto the rocky gradients to then reach all the weights. 
        # It takes the gradients from backward and runs them through Adam's update formula with momentum, average momentum, 
        # adaptive-step-size math.
        self.actor_optimizer.step()

        # We're finding the loss for alpha instead of the actor this time. No critic is involved here at all, we're
        # just using log_probabilitys (probability of choosing an action) and target_entropy(the fixed, desired randomness). 
        # We add log_probabilitys and target_entropy together, producing a gap where negative means the actor is currently MORE 
        # random than desired, positive means LESS random than desired. And we use .detach() to prevent an unintended SECOND 
        # gradient update to the actor's weights. log_probability came from the actor's forward pass, so without .detach(),
        # alpha_loss.backward() would trace gradients all the way back through it into the actor, resulting in the actor 
        # getting nudged by alpha_loss too, on top of its own separate actor_loss update. alpha_loss is only meant to train alpha,
        # not the actor, so .detach() blocks that path. We multiply the gap by log_alpha, since log_alpha is the actual parameter
        # being trained here. A negative gap pushes log_alpha down (less random); a positive gap pushes it up (more random). 
        # .mean() averages this across the whole batch. We then add a negative to this all to minimize loss. So if the inside 
        # is positive meaning more random, it becomes negative and if the inside is negative meaning less random, it makes it 
        # positive to become more random.
        alpha_loss = -(self.log_alpha * (log_probabilitys + self.target_entropy).detach()).mean()

        # This is to get rid of previous gradients that were calculated. PyTorch doesn't automatically clear them, so without this,
        # it'd result in gradients stacking (adding) on top of each other
        self.alpha_optimizer.zero_grad()
        # Backpropogatting to find the slope of each weight and forms a bias, the gradient.
        alpha_loss.backward()
        # After calculating the gradients, PyTorch doesn't automatically begin reading and updating the weights, so this is like
        # running a stream onto the rocky gradients to then reach all the weights. 
        # It takes the gradients from backward and runs them through Adam's update formula with momentum, average momentum, 
        # adaptive-step-size math.
        self.alpha_optimizer.step()

        # This is a one-directional communication, where the source_network gives its parameters to the target network
        # We're then updating the target_critic with the new findings or else it'd be outdated as the actual critic learns.
        # For tau, a percentage that will determine how much the target network's parameters are changed towards the critics 
        # current values. A low value is preferred to help the critic learn without whiplashing from low to high values, so it's 
        # made a small percentage moving towards the critics value
        soft_update(self.critic1, self.target_critic1, self.tau)
        soft_update(self.critic2, self.target_critic2, self.tau)

        # Returning these to track them and see performance of SAC. We need .item() on them all because TensorBoard
        # wants regular floats, not PyTorch Tensor floats attached to the GPU gradients.
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "mean_q1": current_q1.mean().item(),
            "mean_q2": current_q2.mean().item()
        }