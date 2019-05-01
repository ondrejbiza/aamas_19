import argparse
import os
from ...envs.continuous_puck_stack import ContinuousPuckStack

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

    env = ContinuousPuckStack(
        args.size, args.num_pucks, args.goal, args.num_actions, height_noise_std=args.height_noise_std
    )

    while True:

        save_path = None
        if args.save:
            save_path = find_save_path(BASE_PATH)

        x, y = env.show_state(save_path=save_path, draw_grid=args.draw_grid)
        action = int(x) * args.num_actions + int(y)

        if env.hand is not None:
            action += args.num_actions ** 2

        s, r, d, _ = env.step(action)

        print("reward:", r)

        if d:
            if args.save:
                save_path = find_save_path(BASE_PATH)

            env.show_state(save_path=save_path, draw_grid=args.draw_grid)
            env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--num-actions", type=int, default=112)
    parser.add_argument("--num-pucks", type=int, default=3)
    parser.add_argument("--goal", type=int, default=3)

    parser.add_argument("--height-noise-std", type=float, default=0.0)
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--draw-grid", default=False, action="store_true")

    parsed = parser.parse_args()
    main(parsed)
