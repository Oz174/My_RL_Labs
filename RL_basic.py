import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)


class GridWorld:
    def __init__(self, width: int = 5, height: int = 5, start: tuple = (0, 0), goal: tuple = (4, 4), obstacles: list = []):
        self.width = width
        self.height = height
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.state = start
        self.actions = {'up': np.array([-1, 0]),
                        'down': np.array([1, 0]),
                        'left': np.array([0, -1]),
                        'right': np.array([0, 1])}
        self.obs = np.array(
            [obstacle for obstacle in obstacles]) if obstacles else np.array([])

    def reset(self):
        self.state = self.start
        return self.state

    def is_valid_state(self, next_state):
        # the all(obs != next_state).any() is used when the entries of arrays are tuples , comparison is ambiguous if used all(next_states != self.obs) directly
        return (0 <= next_state[0] < self.height) and (0 <= next_state[1] < self.width) and all((next_state != obs).any() for obs in self.obs)

    def step(self, action):
        next_state = self.state + self.actions[action]
        if self.is_valid_state(next_state):
            self.state = next_state
        reward = 100 if (self.state == self.goal).all() else -1
        # compare state.x and goal.x and state.y and goal.y
        # done will be like [true , true] if both are equal
        done = (self.state == self.goal).all()
        return self.state, reward, done


def navigation_policy(env: GridWorld, state: np.array, goal: np.array, obstacles: list):
    actions = ['up', 'down', 'left', 'right']
    valid_actions = {}
    for action in actions:
        next_state = state + env.actions[action]
        if env.is_valid_state(next_state):
            valid_actions[action] = np.sum(np.abs(next_state - goal))
    return min(valid_actions, key=valid_actions.get) if valid_actions else None


def run_simulation_with_policy(env: GridWorld, policy):
    """
    Run the simulation with the given policy

    Args:
    - env: GridWorld environment
    - policy: Policy to be used for navigation
    """
    state = env.reset()
    done = False
    logging.info(f"""Start State: {state}, Goal: {
                 env.goal}, Obstacles: {env.obs}"""
                 )
    while not done:
        # Visualization
        grid = np.zeros((env.height, env.width))
        grid[tuple(state)] = 1  # current state
        grid[tuple(env.goal)] = 2  # goal
        for obstacle in env.obs:
            grid[tuple(obstacle)] = -1  # obstacles

        plt.imshow(grid, cmap='Pastel1')
        # show for 5 seconds then continue executing the code
        plt.pause(0.5)
        # plt.close()

        action = policy(env, state, env.goal, env.obs)
        if action is None:
            logging.info("No valid actions available, agent is stuck.")
            break
        next_state, reward, done = env.step(action)
        logging.info(f"""
                     State:{state} ->
                     Action: {action} ->
                     Next State: {next_state},
                    Reward: {reward}
                    """)
        state = next_state
        if done:
            logging.info("Goal reached!")


if __name__ == "__main__":
    env = GridWorld(
        start=(2, 3),
        obstacles=[(1, 1), (1, 2), (2, 1), (3, 3)]
    )
    run_simulation_with_policy(env, navigation_policy)
