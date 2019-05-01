import os, time
import numpy as np
from .. import vis_utils


class Solver:

    def __init__(self, env, agent, max_time_steps, learning_start, train_freq, max_episodes=None,  rewards_file=None,
                 abstract_actions_file=None, animate=False, animate_from=0, gif_save_path=None, gif_save_limit=None,
                 gif_save_only_successful=False, gif_save_from=None, max_depth_value=3, print_freq=1, train=True):
        """
        Initialize a solver.
        :param env:                             An environment.
        :param agent:                           An agent for the environment.
        :param max_time_steps:                  Maximum time steps to run the agent for.
        :param learning_start:                  Learning starts at this time step.
        :param train_freq:                      Train every N time steps.
        :param max_episodes:                    Maximum episodes to run for.
        :param rewards_file:                    Where to save per-episode rewards.
        :param abstract_actions_file:           Where to save the abstract action taken at each time step.
        :param animate:                         Show an animation of the environment.
        :param animate_from:                    Start the animation after this time step.
        :param gif_save_path:                   Save gifs of episodes.
        :param gif_save_limit:                  Maximum number of gifs to save.
        :param gif_save_only_successful:        Save only gifs of successful episodes.
        :param max_depth_value:                 Maximum depth values; used only for visualization purposes.
        :param print_freq:                      How often to print status.
        :param train:                           Train the agent.
        """

        self.env = env
        self.agent = agent
        self.max_time_steps = max_time_steps
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.max_episodes = max_episodes
        self.rewards_file = rewards_file
        self.abstract_actions_file = abstract_actions_file
        self.animate = animate
        self.animate_from = animate_from
        self.gif_save_path = gif_save_path
        self.gif_save_limit = gif_save_limit
        self.gif_save_only_successful = gif_save_only_successful
        self.gif_save_from = gif_save_from
        self.max_depth_value = max_depth_value
        self.print_freq = print_freq
        self.train = train

        self.episode_rewards = None
        self.actions_to_save = None
        self.animator = None
        self.gif_saver = None
        self.num_saved_gifs = None

        self.reset()

    def reset(self):
        """
        Reset the solver.
        :return:        None.
        """

        # clear lists for rewards and abstract actions
        self.episode_rewards = []
        self.actions_to_save = []

        # initialize new animator
        self.animator = None
        if self.animate:
            self.animator = vis_utils.Animator(0.2, max_value=self.max_depth_value)

        # initialize new gif saver
        self.gif_saver = None
        self.num_saved_gifs = 0
        if self.gif_save_path is not None:
            self.gif_saver = vis_utils.Saver(max_value=self.max_depth_value)

            # maybe create a directory for the gifs
            if not os.path.isdir(self.gif_save_path):
                os.makedirs(self.gif_save_path)

    def run(self):
        """
        Run the solver.
        :return:        None.
        """

        # initialize the environment
        observation = self.env.reset()
        current_episode_rewards = []

        timer_start = time.time()

        # Iterate over time steps
        for t in range(self.max_time_steps):

            # maybe visualize the depth image
            if self.animate and (self.animate_from is None or len(self.episode_rewards) >= self.animate_from):
                self.animator.next(observation[0][:, :, 0])

            if self.gif_saver is not None and \
                    (self.gif_save_limit is None or self.num_saved_gifs < self.gif_save_limit) and \
                    (self.gif_save_from is None or len(self.episode_rewards) >= self.gif_save_from):
                self.gif_saver.add(observation[0][:, :, 0])

            # maybe terminate due to episodes limit
            if self.max_episodes is not None and len(self.episode_rewards) > self.max_episodes:
                break

            # act
            observation, abstract_state, abstract_action, action, new_observation, next_abstract_state, \
                target_abstract_state, success, reward, done = self.agent.act(observation, t)

            # maybe save the abstract action and if the mapping was correct
            if self.abstract_actions_file is not None:
                self.actions_to_save.append([abstract_action, success])

            # learn
            if self.train and self.learning_start is not None and t > self.learning_start and t % self.train_freq == 0:
                self.agent.learn(t)

            # store episode rewards and handle the end of episode
            current_episode_rewards.append(reward)
            if done:

                # maybe show the last observation
                if self.animate and (self.animate_from is None or len(self.episode_rewards) >= self.animate_from):
                    self.animator.next(new_observation[0][:, :, 0])

                # maybe save the last observation
                if self.gif_saver is not None and \
                        (self.gif_save_limit is None or self.num_saved_gifs < self.gif_save_limit) and \
                        (self.gif_save_from is None or len(self.episode_rewards) >= self.gif_save_from):

                    self.gif_saver.add(new_observation[0][:, :, 0])

                    if not self.gif_save_only_successful or reward == 10:
                        self.gif_saver.save_gif(os.path.join(self.gif_save_path, "episode_{:d}.gif".format(len(self.episode_rewards))))
                        self.num_saved_gifs += 1

                    self.gif_saver.reset()

                new_observation = self.env.reset()
                self.episode_rewards.append(np.sum(current_episode_rewards))
                current_episode_rewards = []

            # print status
            if done and self.print_freq is not None and len(self.episode_rewards) % self.print_freq == 0:

                mean_100ep_reward = np.round(np.mean(self.episode_rewards[-50:]), decimals=1)
                num_episodes = len(self.episode_rewards)

                timer_final = time.time()

                exploration_value = self.agent.exploration_schedule.value(t) if self.agent.exploration_schedule is not None else 0

                print("steps: {:d}, episodes: {:d}, mean 100 episode reward: {:.1f}, % time spent exploring {:d}, "
                      "time elapsed {:.2f}".format(t, num_episodes, mean_100ep_reward, int(100 * exploration_value),
                                                   timer_final - timer_start))

                timer_start = timer_final

            observation = new_observation

        # save learning curve
        if self.rewards_file is not None:
            np.savetxt(self.rewards_file, self.episode_rewards)

        # save abstract actions
        if self.abstract_actions_file is not None:
            np.savetxt(self.abstract_actions_file, self.actions_to_save)
