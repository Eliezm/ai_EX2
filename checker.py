
import ex2
import time
import inputs

CODES_NEW = {'passage': 0, 'dragon': 1, 'vault': 2, 'trap': 3, 'hollow_vault': 4, 'vault_trap': 5, 'dragon_trap': 6,
             'hollow_trap_vault': 7}

INVERSE_CODES_NEW = dict([(j, i) for i, j in CODES_NEW.items()])
ACTION_TIMEOUT = 5
CONSTRUCTOR_TIMEOUT = 60
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
INFINITY = 100000


class Checker:
    def __init__(self):
        pass

    def check_controller(self):
        pass

    def true_state_to_controller_input(self):
        pass

    def is_action_legal(self, action):
        pass

    def change_state_after_action(self, action):
        pass

    def at_goal(self):
        pass


class GringottsChecker(Checker):
    game_map: list
    harry_cur_loc: tuple
    turn_limit: int
    dragon_locs: list
    trap_locs: list
    vault_locs: list
    hollow_loc: tuple
    collected_hollow: bool

    def __init__(self, input):
        super().__init__()
        self.game_map = input['full_map']
        self.harry_cur_loc = input['Harry_start']
        self.dragon_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                            if 'dragon' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.trap_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                          if 'trap' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.vault_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                           if 'vault' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.hollow_loc = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                           if 'hollow' in INVERSE_CODES_NEW[self.game_map[x][y]]][0]  # Should only be one
        m = len(self.game_map)
        n = len(self.game_map[0])
        self.turn_limit = 5 + 3 * (n + m)
        self.collected_hollow = False
        print(f"Maximal amount of turns is {self.turn_limit}!")

    def check_controller(self):
        map_dimensions = (len(self.game_map), len(self.game_map[0]))
        observations = self.create_observations()
        constructor_start = time.time()
        controller = ex2.GringottsController(map_dimensions, self.harry_cur_loc, observations)
        constructor_finish = time.time()
        if constructor_finish - constructor_start > CONSTRUCTOR_TIMEOUT:
            return f"Timeout on constructor! Took {constructor_finish - constructor_start} seconds," \
                   f" should take no more than {CONSTRUCTOR_TIMEOUT}"

        counter = 0
        while not self.at_goal():
            observations = self.create_observations()
            start = time.time()
            action = controller.get_next_action(observations)
            finish = time.time()
            if finish - start > ACTION_TIMEOUT:
                return f"Timeout on action! Took {finish - start} seconds, should take no more than {ACTION_TIMEOUT}"
            if not self.is_action_legal(action):
                return f"Action {action} is illegal! Either because the action is impossible or because Harry dies"
            counter += 1
            print(f"Action {action} took {finish - start} seconds!")
            if counter > self.turn_limit:
                return "Turn limit exceeded!"
            self.change_state_after_action(action)
        return f"Goal achieved in {counter} steps!"

    def create_state(self):
        return self.harry_cur_loc

    def create_observations(self):
        observations = []
        close_locs = self.get_close_locs()
        for loc in close_locs:
            if 'dragon' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(('dragon', loc))
            if 'vault' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(('vault', loc))
            if 'trap' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(tuple(['sulfur']))
        observations = list(set(observations))  # Remove duplicates of traps
        return observations

    def is_action_legal(self, action):
        if len(action) == 0 or len(action) >= 3:
            return False
        if len(action) == 1:
            if action[0] == 'wait':
                return True
            if action[0] == 'collect':
                return True
            else:
                return False
        else:
            close_locs = self.get_close_locs()
            if action[0] == 'destroy':
                if action[1] in close_locs:
                    return True
                else:
                    return False
            elif action[0] == 'move':
                new_loc = action[1]
                if new_loc in close_locs:
                    if new_loc in self.dragon_locs:
                        return False
                    if new_loc in self.trap_locs:
                        return False
                    return True
            return False

    def get_close_locs(self):
        harry_y, harry_x = self.harry_cur_loc
        num_rows = len(self.game_map)
        num_cols = len(self.game_map[0])
        return [(harry_y + direction[0], harry_x + direction[1]) for direction in DIRECTIONS
                if ((0 <= harry_y + direction[0] < num_rows) and
                    (0 <= harry_x + direction[1] < num_cols))]

    def change_state_after_action(self, action):
        if action[0] == "move":
            self.change_state_after_moving(action)
        elif action[0] == "destroy":
            self.change_state_after_destroy(action)
        elif action[0] == "collect":
            self.change_state_after_collect()

    def change_state_after_moving(self, action):
        _, loc = action
        self.harry_cur_loc = loc

    def change_state_after_destroy(self, action):
        _, loc = action
        if loc in self.trap_locs:
            self.trap_locs.remove(loc)
            prev_status = INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]
            if prev_status == 'trap':
                new_status = 'passage'
            elif prev_status == 'vault_trap':
                new_status = 'vault'
            elif prev_status == 'dragon_trap':
                new_status = 'dragon'
            else:
                new_status = 'hollow_vault'
            self.game_map[loc[0]][loc[1]] = CODES_NEW[new_status]

    def change_state_after_collect(self):
        if self.harry_cur_loc == self.hollow_loc:
            self.collected_hollow = True

    def at_goal(self):
        return self.collected_hollow


if __name__ == '__main__':
    print(ex2.ids)
    for number, input in enumerate(inputs.easy_inputs):
        my_checker = GringottsChecker(input)
        print(f"Output on input number {number + 1}: {my_checker.check_controller()}")
