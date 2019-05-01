import argparse
import matplotlib.pyplot as plt
from ...envs.continuous_two_stacks import ContinuousTwoStacks


def main(args):

    env = ContinuousTwoStacks(
        args.size, args.num_pucks, args.goal, args.num_actions
    )

    while True:

        x, y = env.show_state()
        action = int(x) * args.num_actions + int(y)

        if env.hand is not None:
            action += args.num_actions ** 2

        s, r, d, _ = env.step(action)

        print("reward:", r)

        if d:
            plt.imshow(s[0][:, :, 0], vmin=0, vmax=args.num_pucks, cmap="gray")
            plt.axis("off")
            plt.show()
            env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--num-actions", type=int, default=112)
    parser.add_argument("--num-pucks", type=int, default=4)
    parser.add_argument("--goal", type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
