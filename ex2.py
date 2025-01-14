# ex2.py

ids = ['123456789']  # <-- Replace with your actual ID(s)

import math
import random
from collections import deque, defaultdict
from heapq import heappush, heappop
from utils import Expr, Symbol, expr, PropKB, pl_resolution, first  # Ensure utils.py is accessible


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

        # Initialize symbol caches
        self.vault_symbols = {}
        self.dragon_symbols = {}
        self.trap_symbols = {}
        self.define_symbols()  # Predefine all symbols

        # Add initial knowledge constraints to the KB
        self.add_knowledge_constraints()

        # Initialize belief dictionaries
        self.Trap_beliefs = {}    # {(r, c): True/False/None}
        self.Dragon_beliefs = {}  # {(r, c): True/False/None}
        self.Vault_beliefs = {}   # {(r, c): True/False/None}

        # Keep track of vaults already "collected" to avoid redundant actions
        self.collected_vaults = set()

        # Store constraints from observations (like "SULFUR+" or "SULFUR0")
        self.obs_constraints = []

        # Track possible traps to destroy
        self.possible_traps_to_destroy = deque()

        # Memory of visited cells
        self.visited = set()
        self.visited.add(harry_loc)  # Add starting location

        # Track path history for inference (last 10 cells)
        self.path_history = deque(maxlen=10)
        self.path_history.append(harry_loc)

        # Define the central point of the grid for heuristic purposes
        self.center = (self.rows // 2, self.cols // 2)

        # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there)
        r0, c0 = harry_loc
        self.kb.tell(~self.trap_symbols[(r0, c0)])
        self.kb.tell(~self.dragon_symbols[(r0, c0)])
        self.Trap_beliefs[(r0, c0)] = False
        self.Dragon_beliefs[(r0, c0)] = False

        # Initialize the goal queue to manage multiple vault targets
        self.goal_queue = deque()
        self.goal_queue_set = set()  # To prevent duplicate additions

        print("----- Initialization -----")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)
        print("--------------------------\n")

        # Incorporate any initial observations
        self.update_with_observations(initial_observations)

        print("----- Post-Initial Observations -----")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)
        print("--------------------------------------\n")

        # Enhanced inference cache: store symbol => Boolean
        self.inference_cache = {}

    # -------------------------------------------------------------------------
    # Knowledge-Base Setup
    # -------------------------------------------------------------------------

    def define_symbols(self):
        """
        Predefine and cache all symbols for Vault, Dragon, and Trap for every cell.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.vault_symbols[(r, c)] = Symbol(f"Vault_{r}_{c}")
                self.dragon_symbols[(r, c)] = Symbol(f"Dragon_{r}_{c}")
                self.trap_symbols[(r, c)] = Symbol(f"Trap_{r}_{c}")

    def cell_symbol(self, kind, r, c):
        """
        Retrieve the cached propositional symbol for a given kind and cell.
        `kind` is a string: "Trap", "Vault", or "Dragon".
        """
        if kind == "Vault":
            return self.vault_symbols.get((r, c))
        elif kind == "Dragon":
            return self.dragon_symbols.get((r, c))
        elif kind == "Trap":
            return self.trap_symbols.get((r, c))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def add_knowledge_constraints(self):
        """
        Add initial knowledge constraints to the KB:
          1. A cell cannot have both a Vault and a Dragon.
          2. If there is a Dragon, then there is no Trap (Dragon => ~Trap).
        """
        exclusivity_clauses = []
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.vault_symbols[(r, c)]
                d = self.dragon_symbols[(r, c)]
                t = self.trap_symbols[(r, c)]

                # 1. Not both Vault and Dragon
                exclusivity_clauses.append(~v | ~d)
                # 2. Dragon implies no Trap (~d ∨ ~t)
                exclusivity_clauses.append(~d | ~t)

        if exclusivity_clauses:
            # Combine all clauses into a single expression in CNF
            clause_str = " & ".join(str(cl) for cl in exclusivity_clauses)
            self.kb.tell(expr(clause_str))

    # -------------------------------------------------------------------------
    # Observations & Updates
    # -------------------------------------------------------------------------

    def update_with_observations(self, obs_list):
        """
        Translate each observation into constraints and add them to the KB.
        """
        print(f"Turn {self.turn_count + 1}: Observations Received: {obs_list}")

        sulfur_detected = False
        sulfur_cells = []

        for obs in obs_list:
            obs_kind = obs[0]
            if obs_kind == "vault":
                (vr, vc) = obs[1]
                v_sym = self.vault_symbols.get((vr, vc))
                d_sym = self.dragon_symbols.get((vr, vc))
                if self.Vault_beliefs.get((vr, vc), None) is not True:
                    self.kb.tell(v_sym)
                    self.kb.tell(~d_sym)
                    self.Vault_beliefs[(vr, vc)] = True
                    self.Dragon_beliefs[(vr, vc)] = False
                    print(f" - Vault detected at {(vr, vc)}. Updated beliefs (Vault=True, Dragon=False).")
                    # Add to goal queue if not already collected
                    if (vr, vc) not in self.collected_vaults and (vr, vc) not in self.goal_queue_set:
                        self.goal_queue.appendleft((vr, vc))  # High priority
                        self.goal_queue_set.add((vr, vc))
                        print(f"   > Added {(vr, vc)} to goal queue.")

            elif obs_kind == "dragon":
                (dr, dc) = obs[1]
                d_sym = self.dragon_symbols.get((dr, dc))
                v_sym = self.vault_symbols.get((dr, dc))
                if self.Dragon_beliefs.get((dr, dc), None) != True:
                    self.kb.tell(d_sym)
                    self.kb.tell(~v_sym)
                    self.Dragon_beliefs[(dr, dc)] = True
                    self.Vault_beliefs[(dr, dc)] = False
                    # By default, set trap false unless we know it's combined
                    self.Trap_beliefs[(dr, dc)] = False
                    self.visited.add((dr, dc))
                    print(f" - Dragon detected at {(dr, dc)}. (Dragon=True, Vault=False, Trap=False)")

            elif obs_kind == "sulfur":
                sulfur_detected = True
                sulfur_cells.append(self.harry_loc)
                print(" - Sulfur detected near Harry's location.")

            else:
                # No specific observation; could log or handle differently
                print(" - Unrecognized observation.")

        # Handle sulfur constraints for current cells
        for cell in sulfur_cells:
            r, c = cell
            neighbors = self.get_4_neighbors(r, c)
            if sulfur_detected:
                # At least one neighbor has a trap: (Trap1 ∨ Trap2 ∨ ...)
                trap_syms = [self.trap_symbols[(nr, nc)] for (nr, nc) in neighbors if
                             (nr, nc) not in self.Dragon_beliefs or not self.Dragon_beliefs.get((nr, nc), False)]
                if trap_syms:
                    sulfur_clause = " | ".join(str(t) for t in trap_syms)
                    self.kb.tell(expr(sulfur_clause))
                    print(f" - Updating KB with sulfur constraint: {sulfur_clause}")
                self.obs_constraints.append(("SULFUR+", cell))
            else:
                # No sulfur detected: ~Trap1 & ~Trap2 & ...
                trap_syms = [~self.trap_symbols[(nr, nc)] for (nr, nc) in neighbors]
                no_trap_clause = " & ".join(str(t) for t in trap_syms)
                self.kb.tell(expr(no_trap_clause))
                print(f" - Updating KB with no sulfur constraint: {no_trap_clause}")
                self.obs_constraints.append(("SULFUR0", cell))

        print()

    def remove_old_sulfur_constraint_for_cell(self, cell):
        """
        Remove old sulfur constraints for a specific cell.
        """
        newlist = [c for c in self.obs_constraints if c[1] != cell]
        self.obs_constraints = newlist

    # -------------------------------------------------------------------------
    # Inference Mechanism
    # -------------------------------------------------------------------------

    def run_inference(self, affected_cells=None):
        """
        Perform inference using the KB to update beliefs.
        Utilizes resolution to deduce new facts.
        """
        cells_to_infer = affected_cells if affected_cells else [
            (r, c) for r in range(self.rows) for c in range(self.cols)
        ]

        for cell in cells_to_infer:
            r, c = cell

            # Vault Inference
            if cell not in self.Vault_beliefs:
                v_sym = self.vault_symbols[(r, c)]
                if v_sym in self.inference_cache:
                    cached_val = self.inference_cache[v_sym]
                    if cached_val:
                        self.Vault_beliefs[cell] = True
                        print(f" - Inference (cached): Vault present at {(r, c)}.")
                    else:
                        self.Vault_beliefs[cell] = False
                        print(f" - Inference (cached): Vault not present at {(r, c)}.")
                    continue
                # Use resolution to check if Vault is present
                result = pl_resolution(self.kb, v_sym)
                self.inference_cache[v_sym] = result
                if result:
                    self.Vault_beliefs[cell] = True
                    print(f" - Inference: Vault present at {(r, c)}.")
                    # Add to goal queue if not already collected
                    if (r, c) not in self.collected_vaults and (r, c) not in self.goal_queue_set:
                        self.goal_queue.appendleft((r, c))  # High priority
                        self.goal_queue_set.add((r, c))
                        print(f"   > Added {(r, c)} to goal queue.")
                    continue
                # Check if Vault is not present
                result_neg = pl_resolution(self.kb, ~v_sym)
                self.inference_cache[~v_sym] = result_neg
                if result_neg:
                    self.Vault_beliefs[cell] = False
                    print(f" - Inference: Vault not present at {(r, c)}.")

            # Dragon Inference
            if cell not in self.Dragon_beliefs:
                d_sym = self.dragon_symbols[(r, c)]
                if d_sym in self.inference_cache:
                    cached_val = self.inference_cache[d_sym]
                    if cached_val:
                        self.Dragon_beliefs[cell] = True
                        print(f" - Inference (cached): Dragon present at {(r, c)}.")
                    else:
                        self.Dragon_beliefs[cell] = False
                        print(f" - Inference (cached): Dragon not present at {(r, c)}.")
                    continue
                # Use resolution to check if Dragon is present
                result = pl_resolution(self.kb, d_sym)
                self.inference_cache[d_sym] = result
                if result:
                    self.Dragon_beliefs[cell] = True
                    print(f" - Inference: Dragon present at {(r, c)}.")
                    continue
                # Check if Dragon is not present
                result_neg = pl_resolution(self.kb, ~d_sym)
                self.inference_cache[~d_sym] = result_neg
                if result_neg:
                    self.Dragon_beliefs[cell] = False
                    print(f" - Inference: Dragon not present at {(r, c)}.")

            # Trap Inference
            if cell not in self.Trap_beliefs:
                t_sym = self.trap_symbols[(r, c)]
                if t_sym in self.inference_cache:
                    cached_val = self.inference_cache[t_sym]
                    if cached_val:
                        self.Trap_beliefs[cell] = True
                        print(f" - Inference (cached): Trap present at {(r, c)}.")
                    else:
                        self.Trap_beliefs[cell] = False
                        print(f" - Inference (cached): Trap not present at {(r, c)}.")
                    continue
                # Use resolution to check if Trap is present
                result = pl_resolution(self.kb, t_sym)
                self.inference_cache[t_sym] = result
                if result:
                    self.Trap_beliefs[cell] = True
                    print(f" - Inference: Trap present at {(r, c)}.")
                    continue
                # Check if Trap is not present
                result_neg = pl_resolution(self.kb, ~t_sym)
                self.inference_cache[~t_sym] = result_neg
                if result_neg:
                    self.Trap_beliefs[cell] = False
                    print(f" - Inference: Trap not present at {(r, c)}.")

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------

    def get_next_action(self, observations):
        """
        Decide on the next action based on current observations and inferred knowledge.
        """
        self.turn_count += 1
        print(f"===== Turn {self.turn_count} =====")
        print(f"Current Location: {self.harry_loc}")

        # 1. Update KB with new observations
        self.update_with_observations(observations)

        # 2. Identify affected cells from observations
        affected_cells = set()
        for obs in observations:
            if obs[0] in ["vault", "dragon", "trap"]:
                affected_cells.add(obs[1])

        if not affected_cells:
            neighbors = self.get_4_neighbors(*self.harry_loc)
            affected_cells.update(neighbors)
            print(f" - No affected cells from observations. Adding neighbors {neighbors} to affected cells.")

        # 3. Run inference to deduce new knowledge
        self.run_inference(affected_cells=affected_cells)

        # 4. Update KB based on path history (to infer safety of adjacent cells)
        self.update_kb_with_path_history()

        # 5. Print debugging info
        self.print_debug_info(label="After Observations & Inference")

        # 6. If sulfur is detected, identify possible traps
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                print(f" - The following cells are suspected as traps: {possible_traps}")
            for trap_cell in possible_traps:
                if (trap_cell not in self.visited) and (self.Trap_beliefs.get(trap_cell, None) is not False):
                    if trap_cell not in self.possible_traps_to_destroy:
                        self.possible_traps_to_destroy.append(trap_cell)
                        print("Possible traps list (appended):", list(self.possible_traps_to_destroy))

        # 7. Prioritize destroying known or suspected traps
        if self.possible_traps_to_destroy:
            trap_to_destroy = self.possible_traps_to_destroy.popleft()

            # Ensure Harry is adjacent to the trap before destroying
            if self.is_adjacent(self.harry_loc, trap_to_destroy):
                action = ("destroy", trap_to_destroy)
                print(f"Action Selected: {action} (Destroying trap at {trap_to_destroy})")
                # Update KB to mark the trap as destroyed
                self.kb.tell(~self.trap_symbols[trap_to_destroy])
                self.Trap_beliefs[trap_to_destroy] = False
                self.visited.add(trap_to_destroy)  # Assuming after destroying, it's safe

                self.print_debug_info(label="After Destroying a Trap")
                print("=============================\n")
                return action
            else:
                # Move closer to the trap using A* for path planning
                path_to_trap = self.a_star_path(self.harry_loc, trap_to_destroy)
                if path_to_trap and len(path_to_trap) > 1:
                    next_step = path_to_trap[1]
                    if self.is_move_safe(next_step):
                        action = ("move", next_step)
                        print(f"Action Selected: {action} (Moving towards trap at {trap_to_destroy})")
                        self.harry_loc = next_step
                        self.visited.add(next_step)
                        self.path_history.append(next_step)
                        self.print_debug_info(label="After Move Action (towards trap)")
                        print("=============================\n")
                        return action

        # 8. If currently on a Vault, collect it
        if self.Vault_beliefs.get(self.harry_loc, None) is True:
            if self.harry_loc not in self.collected_vaults:
                action = ("collect",)
                print(f"Action Selected: {action} (Collecting vault)")
                # Mark the vault as collected
                self.collected_vaults.add(self.harry_loc)
                # Update KB to reflect that the vault has been collected
                self.kb.tell(~self.vault_symbols[self.harry_loc])
                self.Vault_beliefs[self.harry_loc] = False

                self.print_debug_info(label="After Collecting Vault")
                print("=============================\n")
                return action
            else:
                # Already collected this vault; no action needed
                self.Vault_beliefs[self.harry_loc] = False

        # 9. Check for adjacent vaults that also have traps and prioritize destroying traps
        adjacent_vaults_with_traps = [
            cell
            for cell in self.get_4_neighbors(*self.harry_loc)
            if self.Vault_beliefs.get(cell, None) is True
               and self.Trap_beliefs.get(cell, None) is True
        ]
        if adjacent_vaults_with_traps:
            target = adjacent_vaults_with_traps[0]
            action = ("destroy", target)
            print(f"Action Selected: {action} (Destroying trap in adjacent vault at {target})")
            # Update KB to mark the trap as destroyed
            self.kb.tell(~self.trap_symbols[target])
            self.Trap_beliefs[target] = False
            self.visited.add(target)  # Assuming after destroying, it's safe

            self.print_debug_info(label="After Destroying trap in adjacent vault")
            print("=============================\n")
            return action

        # 10. Identify all definite Vaults and plan path to the nearest one
        definite_vaults = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True and (r, c) not in self.collected_vaults:
                    definite_vaults.append((r, c))

        if definite_vaults:
            # Sort definite vaults by Manhattan distance from Harry's current location
            definite_vaults.sort(key=lambda v: abs(v[0] - self.harry_loc[0]) + abs(v[1] - self.harry_loc[1]))
            target_vault = definite_vaults[0]
            # Add to goal queue if not already present
            if target_vault not in self.goal_queue_set:
                self.goal_queue.appendleft(target_vault)  # High priority
                self.goal_queue_set.add(target_vault)
                print(f"   > Added {target_vault} to goal queue.")

        # 11. If sulfur is detected, execute fallback logic to handle possible traps
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                for trap in possible_traps:
                    if trap not in self.visited and self.Trap_beliefs.get(trap, None) is not False:
                        if self.is_adjacent(self.harry_loc, trap):
                            action = ("destroy", trap)
                            print(f"Action Selected: {action} (Destroying fallback sulfur trap)")
                            self.kb.tell(~self.trap_symbols[trap])
                            self.Trap_beliefs[trap] = False
                            self.visited.add(trap)
                            self.print_debug_info(label="After Destroying fallback sulfur trap")
                            print("=============================\n")
                            return action
                        else:
                            # Move closer to the trap with A* path planning
                            path_to_trap = self.a_star_path(self.harry_loc, trap)
                            if path_to_trap and len(path_to_trap) > 1:
                                next_step = path_to_trap[1]
                                if self.is_move_safe(next_step):
                                    action = ("move", next_step)
                                    print(f"Action Selected: {action} (Moving towards fallback trap at {trap})")
                                    self.harry_loc = next_step
                                    self.visited.add(next_step)
                                    self.path_history.append(next_step)
                                    self.print_debug_info(label="After Move Action (towards fallback trap)")
                                    print("=============================\n")
                                    return action

        # 12. Plan path to the most probable vault using enhanced heuristics
        if self.goal_queue:
            target_vault = self.goal_queue.popleft()
            self.goal_queue_set.discard(target_vault)
            path = self.a_star_path(self.harry_loc, target_vault)
            if path and len(path) > 1:
                next_step = path[1]
                if self.is_move_safe(next_step):
                    action = ("move", next_step)
                    print(f"Action Selected: {action} (Moving towards definite vault at {target_vault})")
                    self.harry_loc = next_step
                    self.visited.add(next_step)
                    self.path_history.append(next_step)
                    self.print_debug_info(label="After Move Action (definite vault)")
                    print("=============================\n")
                    return action

        # 13. If no definite goals, prioritize probable vaults
        probable_vaults = [
            (cell, prob) for cell, prob in self.calculate_vault_probabilities().items()
            if prob > 0 and self.Vault_beliefs.get(cell, None) is not True and cell not in self.collected_vaults
        ]
        if probable_vaults:
            # Sort probable vaults by probability descending and distance ascending
            probable_vaults.sort(key=lambda x: (-x[1], abs(x[0][0] - self.harry_loc[0]) + abs(x[0][1] - self.harry_loc[1])))
            target_vault = probable_vaults[0][0]
            if target_vault not in self.goal_queue_set:
                self.goal_queue.append(target_vault)  # Lower priority
                self.goal_queue_set.add(target_vault)
                print(f"   > Added {target_vault} to goal queue as probable vault.")

        if self.goal_queue:
            target_vault = self.goal_queue.popleft()
            self.goal_queue_set.discard(target_vault)
            path = self.a_star_path(self.harry_loc, target_vault)
            if path and len(path) > 1:
                next_step = path[1]
                if self.is_move_safe(next_step):
                    action = ("move", next_step)
                    print(f"Action Selected: {action} (Moving towards probable vault at {target_vault})")
                    self.harry_loc = next_step
                    self.visited.add(next_step)
                    self.path_history.append(next_step)
                    self.print_debug_info(label="After Move Action (probable vault)")
                    print("=============================\n")
                    return action

        # 14. If no goals, explore unvisited safe cells
        path = self.plan_path_to_unvisited_safe()
        if path and len(path) > 1:
            next_step = path[1]
            if self.is_move_safe(next_step):
                action = ("move", next_step)
                print(f"Action Selected: {action} (Exploring safe unvisited cell via A*)")
                self.harry_loc = next_step
                self.visited.add(next_step)
                self.path_history.append(next_step)
                self.print_debug_info(label="After Move Action (exploring safe cell)")
                print("=============================\n")
                return action

        # 15. If no path is found, perform a random move to explore
        random_move = self.get_random_move()
        if random_move:
            action = ("move", random_move)
            print(f"Action Selected: {action} (Performing random move to explore)")
            self.harry_loc = random_move
            self.visited.add(random_move)
            self.path_history.append(random_move)
            self.print_debug_info(label="After Random Move Action")
            print("=============================\n")
            return action

        # 16. Otherwise, wait (should rarely happen)
        action = ("wait",)
        print(f"Action Selected: {action} (No viable action found, waiting)")
        self.print_debug_info(label="After Wait Action")
        print("=============================\n")
        return action

    def is_move_safe(self, cell):
        """
        Check if moving to the specified cell is safe (no known dragon or trap).
        """
        return (self.Dragon_beliefs.get(cell, False) is False and
                self.Trap_beliefs.get(cell, False) is False)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def infer_possible_traps(self):
        """
        If sulfur is detected at some cell, then at least one neighbor is a trap.
        Return all neighbors (not known trap=False) as possible traps.
        """
        possible_traps = set()
        for constraint, cell in self.obs_constraints:
            if constraint == "SULFUR+":
                neighbors = self.get_4_neighbors(cell[0], cell[1])
                for nb in neighbors:
                    if self.Trap_beliefs.get(nb, None) is not False:
                        possible_traps.add(nb)
        return list(possible_traps)

    def plan_path_to_goal(self):
        """
        Plan a path to a definite vault or a probable vault using A* with enhanced heuristics.
        """
        if not self.goal_queue:
            return None
        target_vault = self.goal_queue.popleft()
        self.goal_queue_set.discard(target_vault)
        path = self.a_star_path(self.harry_loc, target_vault)
        if path:
            print(f" - Planned path to goal {target_vault} with enhanced heuristic: {path}")
            return path
        return None

    def plan_path_to_unvisited_safe(self):
        """
        Attempt an A* path to any unvisited safe cell (Trap=False, Dragon=False).
        Return the path if found.
        """
        # We'll pick the unvisited safe cell with minimal Manhattan distance
        safe_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.visited:
                    if self.Trap_beliefs.get((r, c), False) is False and self.Dragon_beliefs.get((r, c), False) is False:
                        dist = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                        safe_cells.append(((r, c), dist))
        if not safe_cells:
            return None
        safe_cells.sort(key=lambda x: x[1])  # sort by distance ascending
        for (target, _) in safe_cells:
            path = self.a_star_path(self.harry_loc, target)
            if path and len(path) > 1:
                return path
        return None

    def a_star_path(self, start, goal):
        """
        A* pathfinding algorithm with enhanced heuristics considering vault probabilities and centrality.
        """
        open_set = []
        heappush(open_set, (self.heuristic(start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            est, cost, current, path = heappop(open_set)
            if current == goal:
                return path
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor in self.get_4_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue
                if self.Trap_beliefs.get(neighbor, False) is True:
                    continue
                if self.Dragon_beliefs.get(neighbor, False) is True:
                    continue
                if neighbor in path:
                    continue
                # Calculate cost and heuristic
                new_cost = cost + 1
                heuristic_value = self.heuristic(neighbor, goal)
                # Incorporate vault probability and centrality into the heuristic
                vault_prob = self.Vault_beliefs.get(neighbor, False)
                centrality = self.heuristic(neighbor, self.center) / (self.rows + self.cols)
                # Adjust combined_est based on heuristic factors
                # Positive vault_prob reduces the heuristic (higher priority)
                # Centrality gives a slight preference to central cells
                if isinstance(vault_prob, bool):
                    vault_bonus = 1.0 if vault_prob else 0.0
                else:
                    vault_bonus = vault_prob  # If probability is a float

                combined_est = new_cost + heuristic_value - (0.5 * vault_bonus) + (0.3 * centrality)
                heappush(open_set, (combined_est, new_cost, neighbor, path + [neighbor]))
        return None

    def get_4_neighbors(self, r, c):
        """
        Return the up/down/left/right neighbors within map boundaries.
        """
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

    def heuristic(self, a, b):
        """
        Simple Manhattan distance for A*.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def calculate_vault_probabilities(self):
        """
        Calculate the probability of each cell containing a vault.
        """
        probabilities = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    probabilities[(r, c)] = 1.0
                elif self.Vault_beliefs.get((r, c), None) is False:
                    probabilities[(r, c)] = 0.0
                else:
                    # Uncertain
                    if (r, c) not in self.visited:
                        neighbors = self.get_4_neighbors(r, c)
                        if any(n in self.visited for n in neighbors):
                            probabilities[(r, c)] = 0.3
                        else:
                            probabilities[(r, c)] = 0.1
                    else:
                        probabilities[(r, c)] = 0.0
        return probabilities

    # -------------------------------------------------------------------------
    # Additional Helper Methods
    # -------------------------------------------------------------------------

    def print_debug_info(self, label=""):
        """
        Print debugging information. Uncomment lines for detailed debugging.
        """
        print(f"--- DEBUG INFO {('[%s]' % label) if label else ''} ---")
        # For deeper debugging, uncomment these lines:
        # print("KB clauses:", self.kb.clauses)
        # print("Trap Beliefs:", self.Trap_beliefs)
        # print("Dragon Beliefs:", self.Dragon_beliefs)
        # print("Vault Beliefs:", self.Vault_beliefs)
        # print("Collected Vaults:", self.collected_vaults)
        # print("Observation Constraints:", self.obs_constraints)
        # print("Possible Traps to Destroy:", list(self.possible_traps_to_destroy))
        # print("Visited Cells:", self.visited)
        # print("Path History:", list(self.path_history))
        print("---------------------------------------\n")

    def update_kb_with_path_history(self):
        """
        Based on the path history, infer that neighboring cells do not contain vaults or traps
        if no observations were made along the path. Exclude cells adjacent to sulfur detections.
        """
        # Identify all cells where sulfur was detected
        sulfur_cells = [cell for constraint, cell in self.obs_constraints if constraint == "SULFUR+"]

        # Identify cells adjacent to any sulfur detection
        sulfur_adjacent = set()
        for cell in sulfur_cells:
            neighbors = self.get_4_neighbors(*cell)
            sulfur_adjacent.update(neighbors)

        for cell in self.path_history:
            neighbors = self.get_4_neighbors(*cell)
            for neighbor in neighbors:
                # Skip cells adjacent to sulfur detections
                if neighbor in sulfur_adjacent:
                    continue
                r, c = neighbor
                # Only infer ~Vault and ~Trap if no observations contradict
                if self.Vault_beliefs.get(neighbor, None) is not True:
                    if (r, c) not in self.visited:
                        # To prevent over-aggressive inference, check if it's safe to infer
                        # Here, we could use multiple inferences or confidence levels
                        # For simplicity, we will skip inferring ~Vault and ~Trap
                        pass
                        # Example:
                        # self.kb.tell(~self.vault_symbols[neighbor])
                        # self.Vault_beliefs[neighbor] = False
                        # print(f" - Inferred no Vault at {neighbor} based on path history.")
                if (self.Trap_beliefs.get(neighbor, None) is not True
                        and not self.Vault_beliefs.get(neighbor, False)):
                    if (r, c) not in self.visited:
                        # Similarly, cautiously avoid inferring ~Trap
                        pass
                        # Example:
                        # self.kb.tell(~self.trap_symbols[neighbor])
                        # self.Trap_beliefs[neighbor] = False
                        # print(f" - Inferred no Trap at {neighbor} based on path history.")

    def is_adjacent(self, cell1, cell2):
        """
        Check if two cells are adjacent (non-diagonal).
        """
        r1, c1 = cell1
        r2, c2 = cell2
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    def get_random_move(self):
        """
        Generate a random valid move from the current location.
        """
        neighbors = self.get_4_neighbors(*self.harry_loc)
        valid_moves = [
            cell for cell in neighbors
            if self.Trap_beliefs.get(cell, False) is not True
            and self.Dragon_beliefs.get(cell, False) is not True
        ]
        # Exclude cells known to be safe but already visited to encourage exploration
        valid_moves = [
            cell for cell in valid_moves
            if not (cell in self.visited and
                    self.Vault_beliefs.get(cell, None) is False
                    and self.Trap_beliefs.get(cell, None) is False)
        ]
        if valid_moves:
            return random.choice(valid_moves)
        return None

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        return "<GringottsController with KB-based inference using PropKB>"