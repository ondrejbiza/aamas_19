import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx


class Graph:

    def __init__(self, nodes, edges, goal, ignore_no_path=False):
        """
        Create a graph.
        :param nodes:               List of nodes in the graph.
        :param edges:               List of edges in the graph represented as tuples.
        :param goal:                Goal node (only one for now).
        :param ignore_no_path:      Don't throw error if there is no path to goal for some node.
        """

        self.nodes = nodes
        self.edges = edges
        self.goal = goal
        self.ignore_no_path = ignore_no_path

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

        self.paths_to_goal = None
        self.lengths_to_goal = None
        self.plan()

    def plan(self):

        paths = nx.all_pairs_shortest_path(self.graph)

        self.paths_to_goal = {}
        self.lengths_to_goal = {}

        for node_paths in paths:

            if self.goal not in node_paths[1]:
                if not self.ignore_no_path:
                    raise ValueError("Path to goal does not exist for some node.")
                else:
                    self.paths_to_goal[node_paths[0]] = None
                    self.lengths_to_goal[node_paths[0]] = None
            else:
                self.paths_to_goal[node_paths[0]] = node_paths[1][self.goal]
                self.lengths_to_goal[node_paths[0]] = len(node_paths[1][self.goal]) - 1

    def get_length(self, node):
        """
        Get length of path to goal.
        :param node:    Start node.
        :return:        Path length.
        """

        return self.lengths_to_goal[node]

    def get_next_step(self, node):
        """
        Get next step towards goal.
        :param node:    Start node.
        :return:        Next node to visit or None, if we are already in goal node.
        """

        if self.lengths_to_goal[node] == 0:
            return None
        else:
            return self.paths_to_goal[node][1]

    def draw(self):

        p = nx.nx_pydot.to_pydot(self.graph)
        png_bytes = p.create_png(prog=["dot", "-Gdpi=500"])

        from io import BytesIO

        sio = BytesIO()
        sio.write(png_bytes)
        sio.seek(0)
        img = mpimg.imread(sio)

        plt.imshow(img, aspect='equal')
        plt.axis("off")
        plt.show()

    def draw_save(self, path):

        p = nx.nx_pydot.to_pydot(self.graph)
        png_bytes = p.create_png(prog=["dot", "-Gdpi=500"])

        with open(path, "wb") as file:
            file.write(png_bytes)
