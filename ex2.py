# ex2.py

ids = ['123456789']  # Replace with your ID(s)

import math
from collections import deque
from utils import Expr, Symbol  # Import Expr and Symbol for logical expressions

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        """
        Controller initialization.
        """
        self.rows, self.cols = map_shape
        self.harry_loc = harry_loc
        self.turn_count = 0

        # Initialize beliefs with uppercase symbols
        self.Trap_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.Dragon_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.Vault_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        # Mark starting cell as definitely not a Trap and not a Dragon
        r0, c0 = harry_loc
        self.Trap_beliefs[r0][c0] = False
        self.Dragon_beliefs[r0][c0] = False

        # Keep track of vaults already "collected" but discovered to be wrong
        self.collected_wrong_vaults = set()

        # Store constraints
        self.obs_constraints = []

        # Incorporate any initial observations
        self.update_with_observations(initial_observations)

        # Queue of planned actions
        self.current_plan = deque()

        # Memory of visited cells
        self.visited = set()
        self.visited.add(harry_loc)  # Add starting location

    def get_next_action(self, observations):
        """
        Decide on the next action with uppercase symbols.
        """
        self.turn_count += 1

        # 1) Update knowledge base with new constraints
        self.update_with_observations(observations)

        # 2) Solve the knowledge base for the most up-to-date assignment
        self.run_inference()

        # Debug: Print current beliefs
        print(f"Turn {self.turn_count}: Current Vault Beliefs: {self.Vault_beliefs}")
        print(f"Turn {self.turn_count}: Current Trap Beliefs: {self.Trap_beliefs}")
        print(f"Turn {self.turn_count}: Current Dragon Beliefs: {self.Dragon_beliefs}")

        # 3) Identify all definite Vaults
        definite_vaults = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs[r][c] is True:
                    definite_vaults.append((r, c))

        # 4) If we are on a Vault, try to collect it
        if (self.harry_loc in definite_vaults) and (self.harry_loc not in self.collected_wrong_vaults):
            action = ("collect",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # 5) If any definite Vault exists and we are not on it, re-plan path to it
        if definite_vaults:
            # For simplicity, choose the first definite Vault found
            target_vault = definite_vaults[0]

            # Plan a path to the target Vault
            path = self.bfs_path(self.harry_loc, target_vault)

            if path and path != [self.harry_loc]:
                # Convert the path to a series of "move" actions
                moves = []
                current = self.harry_loc
                for step in path:
                    if step != current:
                        moves.append(("move", step))
                        current = step
                self.current_plan = deque(moves)

        # 6) If there's a known Trap adjacent, let's destroy it
        destroy_target = self.find_adjacent_definite_trap()
        if destroy_target:
            action = ("destroy", destroy_target)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # 7) If there's a planned action from previous turns, execute it
        if self.current_plan:
            action = self.current_plan.popleft()
            print(f"Turn {self.turn_count}: Action selected: {action}")
            # Update Harry's location if the action is a move
            if action[0] == "move":
                self.harry_loc = action[1]
                self.visited.add(action[1])  # Mark the new cell as visited
            return action

        # 8) Plan a path to a goal (Vault or safe cell)
        path = self.plan_path_to_goal()
        if not path:
            # If no path found, perform a "wait" as a fallback
            action = ("wait",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # Check if the path leads to the current location (implies we are on a goal)
        if path[-1] == self.harry_loc:
            # We are already at the goal, perform "collect" if possible
            if self.Vault_beliefs[r][c] is True:
                action = ("collect",)
                print(f"Turn {self.turn_count}: Action selected: {action}")
                return action

        # Convert the path to a series of "move" actions
        moves = []
        current = self.harry_loc
        for step in path:
            if step != current:
                moves.append(("move", step))
                # Simulate Harry's movement in planning to prevent redundant moves
                current = step
        self.current_plan = deque(moves)

        # Return the next action from the plan
        if self.current_plan:
            action = self.current_plan.popleft()
            print(f"Turn {self.turn_count}: Action selected: {action}")
            # Update Harry's location if the action is a move
            if action[0] == "move":
                self.harry_loc = action[1]
                self.visited.add(action[1])  # Mark the new cell as visited
            return action
        else:
            action = ("wait",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

    # -------------------------------------------------------------------------
    # Observation-based constraints
    # -------------------------------------------------------------------------

    def update_with_observations(self, obs_list):
        """
        Update beliefs based on observations using uppercase symbols.
        """
        # Mark Vault/Dragon if observed
        sulfur_detected = False
        for obs in obs_list:
            if obs[0] == "vault":
                (vr, vc) = obs[1]
                self.Vault_beliefs[vr][vc] = True
                self.Dragon_beliefs[vr][vc] = False
            elif obs[0] == "dragon":
                (dr, dc) = obs[1]
                self.Dragon_beliefs[dr][dc] = True
                self.Vault_beliefs[dr][dc] = False
            elif obs[0] == "sulfur":
                sulfur_detected = True
            elif obs[0] == "trap":
                (tr, tc) = obs[1]
                self.Trap_beliefs[tr][tc] = True

        # Update sulfur constraints
        self.remove_old_sulfur_constraint_for_cell(self.harry_loc)
        if sulfur_detected:
            # sum of Trap among neighbors >= 1
            self.obs_constraints.append(("SULFUR+", self.harry_loc))
        else:
            # sum of Trap among neighbors == 0
            self.obs_constraints.append(("SULFUR0", self.harry_loc))

    def remove_old_sulfur_constraint_for_cell(self, cell):
        """Keep only the most recent sulfur constraint for a cell."""
        newlist = []
        for c in self.obs_constraints:
            ctype, ccell = c
            if ccell != cell:
                newlist.append(c)
        self.obs_constraints = newlist

    # -------------------------------------------------------------------------
    # Logic solver: Backtracking with constraints
    # -------------------------------------------------------------------------

    def run_inference(self):
        """
        Perform backtracking to infer definite truths about the grid.
        """
        # 1) Gather all cells that remain None for Trap or Dragon or Vault
        partial_solution = self._snapshot_current_beliefs()
        cells_to_assign = []
        for r in range(self.rows):
            for c in range(self.cols):
                # Trap
                if self.Trap_beliefs[r][c] is None:
                    cells_to_assign.append(("Trap", r, c))
                # Dragon
                if self.Dragon_beliefs[r][c] is None:
                    cells_to_assign.append(("Dragon", r, c))
                # Vault
                if self.Vault_beliefs[r][c] is None:
                    cells_to_assign.append(("Vault", r, c))

        solutions = []
        max_solutions = 100  # Limit to prevent excessive computation
        self.dpll_backtrack(partial_solution, cells_to_assign, 0, solutions, max_solutions)

        if not solutions:
            # Contradictory observations
            return

        # Compute "forced" booleans = same value in all solutions
        merged = {}
        for var in solutions[0]:
            merged[var] = solutions[0][var]
        for sol in solutions[1:]:
            for var in sol:
                if merged[var] != sol[var]:
                    merged[var] = "UNSURE"

        # Store the definite beliefs
        for (kind, r, c), val in merged.items():
            if val == "UNSURE":
                pass  # remain None
            else:
                # True or False
                if kind == "Trap":
                    self.Trap_beliefs[r][c] = val
                elif kind == "Dragon":
                    self.Dragon_beliefs[r][c] = val
                elif kind == "Vault":
                    self.Vault_beliefs[r][c] = val

    def dpll_backtrack(self, partial_sol, vars_list, index, solutions, max_solutions):
        """
        Depth-first backtracking search to find all consistent assignments.
        Stops after finding max_solutions.
        """
        if len(solutions) >= max_solutions:
            return  # Reached maximum number of solutions

        if index >= len(vars_list):
            # All variables assigned
            solutions.append(dict(partial_sol))
            return

        varinfo = vars_list[index]
        # varinfo = ("Trap"/"Dragon"/"Vault", r, c)
        # If partial_sol already has a value, skip
        if partial_sol[varinfo] is not None:
            if self.check_constraints(partial_sol):
                self.dpll_backtrack(partial_sol, vars_list, index + 1, solutions, max_solutions)
            return
        else:
            for attempt in [False, True]:
                partial_sol[varinfo] = attempt
                if self.check_constraints(partial_sol):
                    self.dpll_backtrack(partial_sol, vars_list, index + 1, solutions, max_solutions)
                if len(solutions) >= max_solutions:
                    return  # Early exit if limit reached
            partial_sol[varinfo] = None  # Reset

    def check_constraints(self, partial_sol):
        """Check if the current partial assignment satisfies all constraints."""
        # 1) Not both Dragon & Vault in the same cell
        for r in range(self.rows):
            for c in range(self.cols):
                dval = partial_sol.get(("Dragon", r, c), None)
                vval = partial_sol.get(("Vault", r, c), None)
                if dval is True and vval is True:
                    return False  # Conflict

        # 2) SULFUR constraints
        for (ctype, cell) in self.obs_constraints:
            (rr, cc) = cell
            neighbors = self.get_4_neighbors(rr, cc)
            trap_sum = 0
            unknown_count = 0
            for (nr, nc) in neighbors:
                tval = partial_sol.get(("Trap", nr, nc), None)
                if tval is True:
                    trap_sum += 1
                elif tval is None:
                    unknown_count += 1

            if ctype == "SULFUR+":
                # At least one Trap
                if trap_sum == 0 and unknown_count == 0:
                    return False
            elif ctype == "SULFUR0":
                # No Traps
                if trap_sum > 0:
                    return False

        # 3) Exactly one Vault exists
        vault_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                vval = partial_sol.get(("Vault", r, c), None)
                if vval is True:
                    vault_count += 1
                elif vval is None:
                    pass  # Potential Vault

        if vault_count > 1:
            return False  # More than one Vault is invalid

        # Additionally, if all cells are assigned and no Vault is present, invalidate
        if vault_count == 0:
            all_assigned = True
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Vault_beliefs[r][c] is None:
                        all_assigned = False
                        break
            if all_assigned:
                return False  # At least one Vault must exist

        return True

    def _snapshot_current_beliefs(self):
        """
        Create a snapshot of current beliefs for backtracking.
        """
        d = {}
        for r in range(self.rows):
            for c in range(self.cols):
                d[("Trap", r, c)] = self.Trap_beliefs[r][c]
                d[("Dragon", r, c)] = self.Dragon_beliefs[r][c]
                d[("Vault", r, c)] = self.Vault_beliefs[r][c]
        return d

    # -------------------------------------------------------------------------
    # Action selection logic
    # -------------------------------------------------------------------------

    def find_adjacent_definite_trap(self):
        """Return the location of an adjacent Trap if known."""
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            tval = self.Trap_beliefs[nr][nc]
            if tval is True:
                return (nr, nc)
        return None

    def calculate_vault_probabilities(self):
        """
        Calculate the probability of each cell containing a vault.
        """
        probabilities = [[0.1 for _ in range(self.cols)] for _ in range(self.rows)]  # Default probability for unknown cells

        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs[r][c] is True:
                    probabilities[r][c] = 1.0
                elif self.Vault_beliefs[r][c] is False:
                    probabilities[r][c] = 0.0
                else:
                    # Adjust probability based on whether the cell has been visited
                    if (r, c) not in self.visited:
                        probabilities[r][c] += 0.1  # Slightly higher probability for unvisited cells
                    # You can further adjust this based on other factors if desired

        return probabilities

    def plan_path_to_goal(self):
        """
        Plan a path to a Vault or a safe cell based on vault probabilities and visited cells.
        """
        # 1. Calculate vault probabilities
        vault_probs = self.calculate_vault_probabilities()

        # 2. Identify candidate goals with their probabilities
        goals = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs[r][c] is True:
                    if (r, c) not in self.collected_wrong_vaults:
                        goals.append(((r, c), 1.0))  # Definite vaults have probability 1
                elif self.Vault_beliefs[r][c] is None:
                    # Prefer unvisited cells by slightly increasing their probability
                    prob = vault_probs[r][c]
                    if (r, c) not in self.visited:
                        prob += 0.1  # Boost probability for unvisited cells
                    goals.append(((r, c), prob))  # Unknown cells have low probability

        # 3. Sort goals by probability in descending order
        goals.sort(key=lambda x: x[1], reverse=True)

        # 4. If no Vaults known, explore safe cells with higher probability
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Trap_beliefs[r][c] is False and self.Dragon_beliefs[r][c] is False:
                        # Prefer unvisited cells by assigning higher probabilities
                        if (r, c) not in self.visited:
                            prob = 0.2  # Higher probability for unvisited safe cells
                        else:
                            prob = 0.1  # Lower probability for visited safe cells
                        goals.append(((r, c), prob))

        # 5. If still no goals, return None
        if not goals:
            return None

        # 6. Select the goal with the highest probability
        best_goal, best_prob = goals[0]

        # 7. Plan a path to the best goal using BFS
        return self.bfs_path(self.harry_loc, best_goal)

    def bfs_path(self, start, goal):
        """
        Perform BFS to find a path from start to goal, avoiding known Traps and Dragons.
        """
        if start == goal:
            return [start]
        from collections import deque
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        while queue:
            (cur, path) = queue.popleft()
            for nbd in self.get_4_neighbors(cur[0], cur[1]):
                nr, nc = nbd
                # Skip if Trap or Dragon is definitely present
                if self.Trap_beliefs[nr][nc] is True:
                    continue
                if self.Dragon_beliefs[nr][nc] is True:
                    continue
                if nbd not in visited:
                    visited.add(nbd)
                    newp = path + [nbd]
                    if nbd == goal:
                        return newp
                    queue.append((nbd, newp))
        return None

    def get_4_neighbors(self, r, c):
        """Return up/down/left/right neighbors within grid boundaries."""
        result = []
        if r > 0:
            result.append((r - 1, c))
        if r < self.rows - 1:
            result.append((r + 1, c))
        if c > 0:
            result.append((r, c - 1))
        if c < self.cols - 1:
            result.append((r, c + 1))
        return result

    def __repr__(self):
        return "<GringottsController using enhanced heuristics for inference and action selection>"
