# ex2.py

ids = ['123456789']  # Replace with your ID(s)

import math
from collections import deque, defaultdict
from utils import Expr, Symbol, expr, PropKB, pl_resolution, first  # Import necessary classes and functions

def cell_symbol(kind, r, c):
    """
    Return a propositional symbol for e.g. 'Trap_0_1' or 'Vault_1_1'.
    `kind` is a string: "Trap", "Vault", or "Dragon".
    """
    return Symbol(f"{kind}_{r}_{c}")

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        """
        Controller initialization using a Knowledge Base (KB) for logical inference.
        """
        self.rows, self.cols = map_shape
        self.harry_loc = harry_loc  # (row, col) with 0-based indexing
        self.turn_count = 0

        # Initialize the Knowledge Base
        self.kb = PropKB()

        # Define Symbols for all cells and add constraints to the KB
        self.define_symbols()
        self.add_knowledge_constraints()

        # Initialize belief dictionaries
        self.Trap_beliefs = {}    # {(r, c): True/False/None}
        self.Dragon_beliefs = {}
        self.Vault_beliefs = {}

        # Keep track of vaults already "collected" but discovered to be wrong
        self.collected_wrong_vaults = set()

        # Store constraints from observations.
        # Constraints are tuples: ("SULFUR+", cell) or ("SULFUR0", cell)
        self.obs_constraints = []

        # Memory of visited cells
        self.visited = set()
        self.visited.add(harry_loc)  # Add starting location

        # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there)
        r0, c0 = harry_loc
        self.kb.tell(~cell_symbol("Trap", r0, c0))
        self.kb.tell(~cell_symbol("Dragon", r0, c0))
        self.Trap_beliefs[(r0, c0)] = False
        self.Dragon_beliefs[(r0, c0)] = False

        print("before:")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)

        # Incorporate any initial observations
        self.update_with_observations(initial_observations)

        print("after:")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)

        # Queue of planned actions
        self.current_plan = deque()

        # Initialize inference cache
        self.inference_cache = {}  # {(cell, kind): True/False}

    def define_symbols(self):
        """
        Define all propositional symbols for Traps, Dragons, and Vaults in the grid.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # Symbols are created on-the-fly using cell_symbol
                pass  # No need to predefine symbols

    def add_knowledge_constraints(self):
        """
        Add initial knowledge constraints to the KB:
          1. A cell cannot have both a Vault and a Dragon.
          2. A cell can have both a Vault and a Trap.
          3. A cell can have both a Dragon and a Trap only in combined tiles.
        """
        exclusivity_clauses = []
        for r in range(self.rows):
            for c in range(self.cols):
                v = cell_symbol("Vault", r, c)
                d = cell_symbol("Dragon", r, c)
                # 1. Not both Vault and Dragon
                exclusivity_clauses.append(~v | ~d)
                # 2. Vault and Trap can coexist (no constraint needed)
                # 3. Dragon and Trap can coexist only if observed as combined tile (handled in observations)

        # Batch tell exclusivity constraints
        if exclusivity_clauses:
            self.kb.tell(expr(" & ".join(str(c) for c in exclusivity_clauses)))

    def update_with_observations(self, obs_list):
        """
        Translate each observation into constraints and add them to the KB.
        """
        # Determine sulfur detection
        sulfur_detected = False
        no_sulfur_detected = False

        for obs in obs_list:
            if obs[0] == "vault":
                (vr, vc) = obs[1]
                v_sym = cell_symbol("Vault", vr, vc)
                d_sym = cell_symbol("Dragon", vr, vc)
                if self.Vault_beliefs.get((vr, vc), None) != True:
                    self.kb.tell(v_sym)
                    self.kb.tell(~d_sym)
                    self.Vault_beliefs[(vr, vc)] = True
                    self.Dragon_beliefs[(vr, vc)] = False
                    self.Trap_beliefs[(vr, vc)] = False  # No trap unless observed
                    self.visited.add((vr, vc))

            elif obs[0] == "hollow_vault":
                (vr, vc) = obs[1]
                hv_sym = cell_symbol("Hollow_Vault", vr, vc)
                v_sym = cell_symbol("Vault", vr, vc)
                t_sym = cell_symbol("Trap", vr, vc)
                if self.Vault_beliefs.get((vr, vc), None) != True:
                    self.kb.tell(v_sym)
                    self.kb.tell(~cell_symbol("Dragon", vr, vc))  # Assuming no dragon unless specified
                    self.kb.tell(t_sym)
                    self.Vault_beliefs[(vr, vc)] = True
                    self.Trap_beliefs[(vr, vc)] = True
                    self.visited.add((vr, vc))

            elif obs[0] == "dragon":
                (dr, dc) = obs[1]
                d_sym = cell_symbol("Dragon", dr, dc)
                v_sym = cell_symbol("Vault", dr, dc)
                if self.Dragon_beliefs.get((dr, dc), None) != True:
                    self.kb.tell(d_sym)
                    self.kb.tell(~v_sym)
                    self.Dragon_beliefs[(dr, dc)] = True
                    self.Vault_beliefs[(dr, dc)] = False
                    # **Important Fix:** Set Trap Beliefs to False unless combined tile is observed
                    self.Trap_beliefs[(dr, dc)] = False
                    self.visited.add((dr, dc))

            elif obs[0] == "trap":
                (tr, tc) = obs[1]
                t_sym = cell_symbol("Trap", tr, tc)
                if self.Trap_beliefs.get((tr, tc), None) != True:
                    self.kb.tell(t_sym)
                    self.Trap_beliefs[(tr, tc)] = True
                    self.visited.add((tr, tc))

            elif obs[0] == "sulfur":
                sulfur_detected = True

            elif obs[0] == "no_sulfur":
                no_sulfur_detected = True

        # Handle sulfur constraints for the current cell
        r, c = self.harry_loc
        neighbors = self.get_4_neighbors(r, c)
        neighbor_traps = [cell_symbol("Trap", nr, nc) for (nr, nc) in neighbors]

        # Remove old sulfur constraints for this cell
        self.remove_old_sulfur_constraint_for_cell(self.harry_loc)

        if sulfur_detected:
            # At least one neighbor Trap: (Trap_n1 | Trap_n2 | ...)
            if neighbor_traps:
                self.kb.tell(expr(" | ".join(str(t) for t in neighbor_traps)))
            self.obs_constraints.append(("SULFUR+", self.harry_loc))
        elif no_sulfur_detected:
            # No neighbor Traps: (~Trap_n1) & (~Trap_n2) & ...
            if neighbor_traps:
                no_trap_clauses = " & ".join(str(~t) for t in neighbor_traps)
                self.kb.tell(expr(no_trap_clauses))
            self.obs_constraints.append(("SULFUR0", self.harry_loc))

    def remove_old_sulfur_constraint_for_cell(self, cell):
        """
        Keep only the most recent sulfur constraint for the given cell in obs_constraints.
        """
        newlist = [c for c in self.obs_constraints if c[1] != cell]
        self.obs_constraints = newlist

    def run_inference(self, affected_cells=None):
        """
        Perform inference using the KB to update belief dictionaries.
        If affected_cells is provided, only infer for those cells.
        """
        cells_to_infer = affected_cells if affected_cells else [(r, c) for r in range(self.rows) for c in range(self.cols)]
        for cell in cells_to_infer:
            r, c = cell
            # Infer Vaults
            if cell not in self.Vault_beliefs:
                v_sym = cell_symbol("Vault", r, c)
                cache_key = (cell, "Vault")
                if cache_key in self.inference_cache:
                    inference_result = self.inference_cache[cache_key]
                else:
                    inference_result = pl_resolution(self.kb, v_sym)
                    self.inference_cache[cache_key] = inference_result
                if inference_result:
                    self.Vault_beliefs[cell] = True
                    continue  # No need to check False if True

                cache_key_neg = (cell, "Vault_neg")
                if cache_key_neg in self.inference_cache:
                    inference_result_neg = self.inference_cache[cache_key_neg]
                else:
                    inference_result_neg = pl_resolution(self.kb, ~v_sym)
                    self.inference_cache[cache_key_neg] = inference_result_neg
                if inference_result_neg:
                    self.Vault_beliefs[cell] = False

            # Infer Dragons
            if cell not in self.Dragon_beliefs:
                d_sym = cell_symbol("Dragon", r, c)
                cache_key = (cell, "Dragon")
                if cache_key in self.inference_cache:
                    inference_result = self.inference_cache[cache_key]
                else:
                    inference_result = pl_resolution(self.kb, d_sym)
                    self.inference_cache[cache_key] = inference_result
                if inference_result:
                    self.Dragon_beliefs[cell] = True
                    continue

                cache_key_neg = (cell, "Dragon_neg")
                if cache_key_neg in self.inference_cache:
                    inference_result_neg = self.inference_cache[cache_key_neg]
                else:
                    inference_result_neg = pl_resolution(self.kb, ~d_sym)
                    self.inference_cache[cache_key_neg] = inference_result_neg
                if inference_result_neg:
                    self.Dragon_beliefs[cell] = False

            # Infer Traps
            if cell not in self.Trap_beliefs:
                t_sym = cell_symbol("Trap", r, c)
                cache_key = (cell, "Trap")
                if cache_key in self.inference_cache:
                    inference_result = self.inference_cache[cache_key]
                else:
                    inference_result = pl_resolution(self.kb, t_sym)
                    self.inference_cache[cache_key] = inference_result
                if inference_result:
                    self.Trap_beliefs[cell] = True
                    continue

                cache_key_neg = (cell, "Trap_neg")
                if cache_key_neg in self.inference_cache:
                    inference_result_neg = self.inference_cache[cache_key_neg]
                else:
                    inference_result_neg = pl_resolution(self.kb, ~t_sym)
                    self.inference_cache[cache_key_neg] = inference_result_neg
                if inference_result_neg:
                    self.Trap_beliefs[cell] = False

    def get_next_action(self, observations):
        """
        Decide on the next action based on current observations and inferred knowledge.
        """
        self.turn_count += 1

        # 1. Update KB with new observations
        self.update_with_observations(observations)

        # 2. Identify affected cells from observations
        affected_cells = set()
        for obs in observations:
            if obs[0] in ["vault", "dragon", "trap", "hollow_vault"]:
                affected_cells.add(obs[1])

        # 3. Run inference on affected cells
        self.run_inference(affected_cells=affected_cells)

        # 4. Print current beliefs for debugging
        self.print_beliefs()

        # 5. Check if current location is a Vault
        if self.Vault_beliefs.get(self.harry_loc, None) is True:
            if self.Trap_beliefs.get(self.harry_loc, None) is True:
                # Trap is present in the vault cell; prioritize destroying it
                action = ("destroy", self.harry_loc)
                print(f"Turn {self.turn_count}: Action selected: {action}")
                # Update KB: Destroyed trap implies no trap in that cell
                self.kb.tell(~cell_symbol("Trap", self.harry_loc[0], self.harry_loc[1]))
                self.Trap_beliefs[self.harry_loc] = False
                return action
            else:
                # No trap present; collect the vault
                action = ("collect",)
                print(f"Turn {self.turn_count}: Action selected: {action}")
                return action

        # 6. Check for any adjacent definite Traps and destroy them
        destroy_target = self.find_adjacent_definite_trap()
        if destroy_target:
            action = ("destroy", destroy_target)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            # Update KB: Destroyed trap implies no trap in that cell
            self.kb.tell(~cell_symbol("Trap", destroy_target[0], destroy_target[1]))
            self.Trap_beliefs[destroy_target] = False
            return action

        # 7. Identify all definite Vaults
        definite_vaults = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    definite_vaults.append((r, c))

        # 8. If any definite Vault exists, plan a path to it
        if definite_vaults:
            # Choose the closest definite Vault
            target_vault = self.get_closest_vault(definite_vaults)
            path = self.a_star_path(self.harry_loc, target_vault)
            if path and path != [self.harry_loc]:
                # Move to the first step in the path
                next_step = path[1]
                action = ("move", next_step)
                print(f"Turn {self.turn_count}: Action selected: {action}")
                self.harry_loc = next_step
                self.visited.add(next_step)
                return action

        # 9. If sulfur is detected, infer and destroy possible traps
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            # Infer possible trap locations
            possible_traps = self.infer_possible_traps()
            for trap in possible_traps:
                if trap not in self.visited and self.Trap_beliefs.get(trap, None) is not False:
                    action = ("destroy", trap)
                    print(f"Turn {self.turn_count}: Action selected: {action}")
                    # Update KB: Destroyed trap implies no trap in that cell
                    self.kb.tell(~cell_symbol("Trap", trap[0], trap[1]))
                    self.Trap_beliefs[trap] = False
                    return action

        # 10. If no definite Vaults, plan a path to the most probable Vault
        path = self.plan_path_to_goal()
        if path and path != [self.harry_loc]:
            next_step = path[1]
            action = ("move", next_step)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            self.harry_loc = next_step
            self.visited.add(next_step)
            return action

        # 11. If no path found, wait
        action = ("wait",)
        print(f"Turn {self.turn_count}: Action selected: {action}")
        return action

    def get_closest_vault(self, definite_vaults):
        """
        Return the closest definite Vault based on Manhattan distance.
        """
        min_distance = math.inf
        closest_vault = None
        for vault in definite_vaults:
            distance = abs(vault[0] - self.harry_loc[0]) + abs(vault[1] - self.harry_loc[1])
            if distance < min_distance:
                min_distance = distance
                closest_vault = vault
        return closest_vault

    # -------------------------------------------------------------------------
    # Action selection logic helpers
    # -------------------------------------------------------------------------

    def find_adjacent_definite_trap(self):
        """Return the location of an adjacent Trap if known = True."""
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            if self.Trap_beliefs.get((nr, nc), None) is True:
                return (nr, nc)
        return None

    def infer_possible_traps(self):
        """
        Infer possible trap locations based on current sulfur constraints and beliefs.
        Returns a list of cells that might contain traps.
        """
        possible_traps = set()
        for constraint, cell in self.obs_constraints:
            if constraint == "SULFUR+":
                neighbors = self.get_4_neighbors(cell[0], cell[1])
                for neighbor in neighbors:
                    if self.Trap_beliefs.get(neighbor, None) is not False:
                        possible_traps.add(neighbor)
        return list(possible_traps)

    def calculate_vault_probabilities(self):
        """
        Calculate a refined probability for each cell containing a vault,
        based on current knowledge and proximity to visited cells.
        """
        probabilities = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    probabilities[(r, c)] = 1.0
                elif self.Vault_beliefs.get((r, c), None) is False:
                    probabilities[(r, c)] = 0.0
                else:
                    # Higher probability if adjacent to visited cells
                    if (r, c) not in self.visited:
                        neighbors = self.get_4_neighbors(r, c)
                        adjacent_visited = any(n in self.visited for n in neighbors)
                        if adjacent_visited:
                            probabilities[(r, c)] = 0.3  # Increased probability
                        else:
                            probabilities[(r, c)] = 0.1  # Base probability
                    else:
                        probabilities[(r, c)] = 0.0  # Already visited
        return probabilities

    def plan_path_to_goal(self):
        """
        Plan a path to a Vault or a safe cell, using A* with combined probability and distance heuristics.
        """
        vault_probs = self.calculate_vault_probabilities()

        # Collect candidate goals
        goals = []
        # 1. Definite Vaults
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    goals.append(((r, c), vault_probs[(r, c)]))

        # 2. Probable Vaults
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Vault_beliefs.get((r, c), None) is None and vault_probs.get((r, c), 0) > 0:
                        distance = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                        # Combine probability and inverse distance
                        combined_score = vault_probs[(r, c)] / (distance + 1)  # +1 to avoid division by zero
                        goals.append(((r, c), combined_score))

        # 3. Safe Cells (No known traps or dragons)
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Trap_beliefs.get((r, c), False) is False and self.Dragon_beliefs.get((r, c), False) is False:
                        if (r, c) not in self.visited:
                            distance = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                            combined_score = 0.2 / (distance + 1)
                            goals.append(((r, c), combined_score))

        if not goals:
            return None

        # Sort goals by combined_score in descending order
        goals.sort(key=lambda x: x[1], reverse=True)

        # Select the best goal
        best_goal, _ = goals[0]
        path = self.a_star_path(self.harry_loc, best_goal)
        if path:
            return path
        else:
            # If A* fails, fall back to BFS
            return self.bfs_path(self.harry_loc, best_goal)

    def a_star_path(self, start, goal):
        """
        A* pathfinding avoiding known traps/dragons and inferred possible traps.
        Heuristic: Manhattan distance to the goal.
        """
        from heapq import heappush, heappop

        open_set = []
        heappush(open_set, (0 + self.heuristic(start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            estimated_total, cost, current, path = heappop(open_set)
            if current == goal:
                return path
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor in self.get_4_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue
                # Avoid known traps and dragons
                if self.Trap_beliefs.get(neighbor, None) is True:
                    continue
                if self.Dragon_beliefs.get(neighbor, None) is True:
                    continue
                # Avoid inferred possible traps unless they're being destroyed
                if any(neighbor in self.infer_possible_traps() for _ in [0]):
                    continue
                new_cost = cost + 1
                estimated = new_cost + self.heuristic(neighbor, goal)
                heappush(open_set, (estimated, new_cost, neighbor, path + [neighbor]))
        return None

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs_path(self, start, goal):
        """
        BFS path avoiding known traps/dragons.
        """
        if start == goal:
            return [start]
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        while queue:
            (current, path) = queue.popleft()
            for nbd in self.get_4_neighbors(current[0], current[1]):
                nr, nc = nbd
                # Skip if definitely a trap or definitely a dragon
                if self.Trap_beliefs.get((nr, nc), None) is True:
                    continue
                if self.Dragon_beliefs.get((nr, nc), None) is True:
                    continue
                if nbd not in visited:
                    visited.add(nbd)
                    new_path = path + [nbd]
                    if nbd == goal:
                        return new_path
                    queue.append((nbd, new_path))
        return None

    def get_4_neighbors(self, r, c):
        """Return up/down/left/right neighbors within grid boundaries."""
        results = []
        if r > 0:
            results.append((r - 1, c))
        if r < self.rows - 1:
            results.append((r + 1, c))
        if c > 0:
            results.append((r, c - 1))
        if c < self.cols - 1:
            results.append((r, c + 1))
        return results

    def find_adjacent_definite_trap(self):
        """Return the location of an adjacent Trap if known = True."""
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            if self.Trap_beliefs.get((nr, nc), None) is True:
                return (nr, nc)
        return None

    def infer_possible_traps(self):
        """
        Infer possible trap locations based on current sulfur constraints and beliefs.
        Returns a list of cells that might contain traps.
        """
        possible_traps = set()
        for constraint, cell in self.obs_constraints:
            if constraint == "SULFUR+":
                neighbors = self.get_4_neighbors(cell[0], cell[1])
                for neighbor in neighbors:
                    if self.Trap_beliefs.get(neighbor, None) is not False:
                        possible_traps.add(neighbor)
        return list(possible_traps)

    def calculate_vault_probabilities(self):
        """
        Calculate a refined probability for each cell containing a vault,
        based on current knowledge and proximity to visited cells.
        """
        probabilities = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    probabilities[(r, c)] = 1.0
                elif self.Vault_beliefs.get((r, c), None) is False:
                    probabilities[(r, c)] = 0.0
                else:
                    # Higher probability if adjacent to visited cells
                    if (r, c) not in self.visited:
                        neighbors = self.get_4_neighbors(r, c)
                        adjacent_visited = any(n in self.visited for n in neighbors)
                        if adjacent_visited:
                            probabilities[(r, c)] = 0.3  # Increased probability
                        else:
                            probabilities[(r, c)] = 0.1  # Base probability
                    else:
                        probabilities[(r, c)] = 0.0  # Already visited
        return probabilities

    def plan_path_to_goal(self):
        """
        Plan a path to a Vault or a safe cell, using A* with combined probability and distance heuristics.
        """
        vault_probs = self.calculate_vault_probabilities()

        # Collect candidate goals
        goals = []
        # 1. Definite Vaults
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    goals.append(((r, c), vault_probs[(r, c)]))

        # 2. Probable Vaults
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Vault_beliefs.get((r, c), None) is None and vault_probs.get((r, c), 0) > 0:
                        distance = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                        # Combine probability and inverse distance
                        combined_score = vault_probs[(r, c)] / (distance + 1)  # +1 to avoid division by zero
                        goals.append(((r, c), combined_score))

        # 3. Safe Cells (No known traps or dragons)
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Trap_beliefs.get((r, c), False) is False and self.Dragon_beliefs.get((r, c), False) is False:
                        if (r, c) not in self.visited:
                            distance = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                            combined_score = 0.2 / (distance + 1)
                            goals.append(((r, c), combined_score))

        if not goals:
            return None

        # Sort goals by combined_score in descending order
        goals.sort(key=lambda x: x[1], reverse=True)

        # Select the best goal
        best_goal, _ = goals[0]
        path = self.a_star_path(self.harry_loc, best_goal)
        if path:
            return path
        else:
            # If A* fails, fall back to BFS
            return self.bfs_path(self.harry_loc, best_goal)

    def a_star_path(self, start, goal):
        """
        A* pathfinding avoiding known traps/dragons and inferred possible traps.
        Heuristic: Manhattan distance to the goal.
        """
        from heapq import heappush, heappop

        open_set = []
        heappush(open_set, (0 + self.heuristic(start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            estimated_total, cost, current, path = heappop(open_set)
            if current == goal:
                return path
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor in self.get_4_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue
                # Avoid known traps and dragons
                if self.Trap_beliefs.get(neighbor, None) is True:
                    continue
                if self.Dragon_beliefs.get(neighbor, None) is True:
                    continue
                # Avoid inferred possible traps unless they're being destroyed
                if any(neighbor in self.infer_possible_traps() for _ in [0]):
                    continue
                new_cost = cost + 1
                estimated = new_cost + self.heuristic(neighbor, goal)
                heappush(open_set, (estimated, new_cost, neighbor, path + [neighbor]))
        return None

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs_path(self, start, goal):
        """
        BFS path avoiding known traps/dragons.
        """
        if start == goal:
            return [start]
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        while queue:
            (current, path) = queue.popleft()
            for nbd in self.get_4_neighbors(current[0], current[1]):
                nr, nc = nbd
                # Skip if definitely a trap or definitely a dragon
                if self.Trap_beliefs.get((nr, nc), None) is True:
                    continue
                if self.Dragon_beliefs.get((nr, nc), None) is True:
                    continue
                if nbd not in visited:
                    visited.add(nbd)
                    new_path = path + [nbd]
                    if nbd == goal:
                        return new_path
                    queue.append((nbd, new_path))
        return None

    def get_4_neighbors(self, r, c):
        """Return up/down/left/right neighbors within grid boundaries."""
        results = []
        if r > 0:
            results.append((r - 1, c))
        if r < self.rows - 1:
            results.append((r + 1, c))
        if c > 0:
            results.append((r, c - 1))
        if c < self.cols - 1:
            results.append((r, c + 1))
        return results

    def find_adjacent_definite_trap(self):
        """Return the location of an adjacent Trap if known = True."""
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            if self.Trap_beliefs.get((nr, nc), None) is True:
                return (nr, nc)
        return None

    def infer_possible_traps(self):
        """
        Infer possible trap locations based on current sulfur constraints and beliefs.
        Returns a list of cells that might contain traps.
        """
        possible_traps = set()
        for constraint, cell in self.obs_constraints:
            if constraint == "SULFUR+":
                neighbors = self.get_4_neighbors(cell[0], cell[1])
                for neighbor in neighbors:
                    if self.Trap_beliefs.get(neighbor, None) is not False:
                        possible_traps.add(neighbor)
        return list(possible_traps)

    def calculate_vault_probabilities(self):
        """
        Calculate a refined probability for each cell containing a vault,
        based on current knowledge and proximity to visited cells.
        """
        probabilities = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    probabilities[(r, c)] = 1.0
                elif self.Vault_beliefs.get((r, c), None) is False:
                    probabilities[(r, c)] = 0.0
                else:
                    # Higher probability if adjacent to visited cells
                    if (r, c) not in self.visited:
                        neighbors = self.get_4_neighbors(r, c)
                        adjacent_visited = any(n in self.visited for n in neighbors)
                        if adjacent_visited:
                            probabilities[(r, c)] = 0.3  # Increased probability
                        else:
                            probabilities[(r, c)] = 0.1  # Base probability
                    else:
                        probabilities[(r, c)] = 0.0  # Already visited
        return probabilities

    def print_beliefs(self):
        """Helper method to print current beliefs."""
        print(f"Turn {self.turn_count}: Current Vault Beliefs:")
        for r in range(self.rows):
            row_beliefs = []
            for c in range(self.cols):
                belief = self.Vault_beliefs.get((r, c), None)
                row_beliefs.append(belief)
            print(row_beliefs)
        print(f"Turn {self.turn_count}: Current Trap Beliefs:")
        for r in range(self.rows):
            row_beliefs = []
            for c in range(self.cols):
                belief = self.Trap_beliefs.get((r, c), None)
                row_beliefs.append(belief)
            print(row_beliefs)
        print(f"Turn {self.turn_count}: Current Dragon Beliefs:")
        for r in range(self.rows):
            row_beliefs = []
            for c in range(self.cols):
                belief = self.Dragon_beliefs.get((r, c), None)
                row_beliefs.append(belief)
            print(row_beliefs)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        return "<GringottsController with KB-based inference using PropKB>"
