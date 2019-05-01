import argparse
import os
import matplotlib.pyplot as plt
from ...envs.continuous_stairs import ContinuousStairs

BASE_PATH = "."


def find_save_path(base_path):

    step = 0
    while True:
        path = os.path.join(base_path, "{:d}.png".format(step))
        if not os.path.isfile(path):
            break
        step += 1

    return path


def main(args):

    if args.size == args.num_actions:
        tolerance = 14
    else:
        tolerance = 8

    env = ContinuousStairs(
        args.size, args.num_pucks, args.goal, args.num_actions, tolerance=tolerance
    )

    while True:

        save_path = None
        if args.save:
            save_path = find_save_path(BASE_PATH)

        x, y = env.show_state(save_path=save_path)
        action = int(x) * args.num_actions + int(y)

        if env.hand is not None:
            action += args.num_actions ** 2

        s, r, d, _ = env.step(action)

        print("reward:", r)

        if d:
            plt.imshow(s[0][:, :, 0], vmin=0, vmax=args.num_pucks, cmap="gray")
            plt.axis("off")

            if args.save:
                plt.imsave(find_save_path(BASE_PATH), s[0][:, :, 0], vmin=0, vmax=args.num_pucks, cmap="gray")

            plt.show()
            env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--num-actions", type=int, default=112)
    parser.add_argument("--num-pucks", type=int, default=3)
    parser.add_argument("--goal", type=int, default=3)

    parser.add_argument("--save", default=False, action="store_true")

    parsed = parser.parse_args()
    main(parsed)
