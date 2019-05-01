from .continuous_puck_stack import ContinuousPuckStack


class ContinuousComponent(ContinuousPuckStack):

    def __init__(self, num_blocks, num_pucks, goal, num_actions, min_radius=7, max_radius=12, tolerance=8,
                 no_contact=True):

        self.tolerance = tolerance

        ContinuousPuckStack.__init__(
            self, num_blocks, num_pucks, goal, num_actions, min_radius=min_radius, max_radius=max_radius,
            no_contact=no_contact
        )

    def check_goal(self):

        if len(self.layers[0]) < self.goal:
            return False

        all_pucks = set(self.layers[0])

        while len(all_pucks) > 0:

            # go over all pucks
            component = {all_pucks.pop()}
            change = True

            while change:

                change = False

                for puck_1 in component:

                    for puck_2 in all_pucks:

                        dist = self.euclid(puck_1, [puck_2.x, puck_2.y])

                        form_component = dist <= puck_1.radius + puck_2.radius + self.tolerance

                        if form_component and puck_2 not in component:
                            # this puck forms a component with the seed, no need to check it again
                            component.add(puck_2)
                            all_pucks.remove(puck_2)
                            change = True
                            break

                    if change:
                        break

            if len(component) >= self.goal:
                return True

        return False
