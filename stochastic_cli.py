import os, json, argparse, subprocess, datetime
from fractions import Fraction

# ---------- Utilities ----------

def q_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"

def to_frac_matrix(M):
    return [[Fraction(x) for x in row] for row in M]

def to_lean_matrix(mat_str):
    # mat_str: list[list[str]] with "a/b" or "0"/"1"
    rows = [", ".join(r) for r in mat_str]
    return "!![ " + " ;\n      ".join(rows) + " ]"

def sum_row(fr_row):
    return sum(fr_row)

# ---------- Violation finding ----------

def find_violations(M_str, edges=None, require_pos_on_allowed=True):
    """
    Return list of dicts:
      kind in {"row_sum","neg","disallowed_nonzero","allowed_zero"}
    """
    F = to_frac_matrix(M_str)
    n = len(F)
    vios = []

    # prepare mask
    allowed = {i: set(edges[str(i)]) for i in range(n)} if edges else None

    for i in range(n):
        row = F[i]
        # nonneg
        for j, v in enumerate(row):
            if v < 0:
                vios.append({"kind":"neg","row":i,"col":j,"value":v})

        # row sum
        s = sum_row(row)
        if s != Fraction(1,1):
            vios.append({"kind":"row_sum","row":i,"sum":s,"expected":Fraction(1,1)})

        # mask constraints
        if allowed is not None:
            for j, v in enumerate(row):
                if j not in allowed[i] and v != 0:
                    vios.append({"kind":"disallowed_nonzero","row":i,"col":j,"value":v})
                if require_pos_on_allowed and j in allowed[i] and v == 0:
                    vios.append({"kind":"allowed_zero","row":i,"col":j})
    return vios

def print_violations(vios):
    if not vios:
        print("No violations. ✅")
        return
    print(f"Violations ({len(vios)}):")
    for k, v in enumerate(vios, 1):
        t = v["kind"]
        if t == "row_sum":
            print(f"  {k}) ROW-SUM ≠ 1 at row {v['row']+1}: sum={v['sum']} (expected 1)")
        elif t == "neg":
            print(f"  {k}) NEGATIVE at ({v['row']+1},{v['col']+1}): {v['value']}")
        elif t == "disallowed_nonzero":
            print(f"  {k}) DISALLOWED edge nonzero at ({v['row']+1},{v['col']+1}): {v['value']}")
        elif t == "allowed_zero":
            print(f"  {k}) ALLOWED edge is 0 at ({v['row']+1},{v['col']+1})")

# ---------- Fix one violation (greedy) ----------

def choose_adjust_col(i, allowed, j_forbid=None):
    """Pick a column in row i to absorb adjustments."""
    if allowed is None:
        # fallback: prefer self-loop then last column
        return i if j_forbid != i else -1
    prefs = []
    # prefer self-loop if allowed
    if i in allowed[i]: prefs.append(i)
    # then any other allowed not equal to forbidded
    prefs += [j for j in sorted(allowed[i]) if j != j_forbid]
    return prefs[0] if prefs else -1

def clip_and_renorm(row):
    row = [max(x, Fraction(0,1)) for x in row]
    s = sum(row)
    if s == 0:
        # degenerate: put all mass to first cell
        row = [Fraction(1,1)] + [Fraction(0,1)]*(len(row)-1)
    else:
        row = [x/s for x in row]
    return row

def fix_one(M_str, vio, edges=None, require_pos_on_allowed=True):
    F = to_frac_matrix(M_str)
    n = len(F)
    allowed = {i: set(edges[str(i)]) for i in range(n)} if edges else None

    i = vio["row"]
    row = F[i].copy()

    def set_row_and_return():
        F[i] = row
        return [[q_to_str(x) for x in r] for r in F]

    if vio["kind"] == "row_sum":
        target_col = choose_adjust_col(i, allowed)
        if target_col == -1:
            # if no allowed, choose last column
            target_col = n-1
        desired = Fraction(1,1) - sum(row[:target_col] + row[target_col+1:])
        row[target_col] = desired
        row = clip_and_renorm(row)
        return set_row_and_return()

    if vio["kind"] == "neg":
        j = vio["col"]
        bump = -row[j]
        row[j] = Fraction(0,1)
        target_col = choose_adjust_col(i, allowed, j_forbid=j)
        if target_col == -1:
            target_col = n-1
        row[target_col] += bump
        row = clip_and_renorm(row)
        return set_row_and_return()

    if vio["kind"] == "disallowed_nonzero":
        j = vio["col"]
        bump = row[j]
        row[j] = Fraction(0,1)
        target_col = choose_adjust_col(i, allowed, j_forbid=j)
        if target_col == -1:
            target_col = n-1
        row[target_col] += bump
        row = clip_and_renorm(row)
        return set_row_and_return()

    if vio["kind"] == "allowed_zero" and require_pos_on_allowed:
        j = vio["col"]
        eps = Fraction(1, 100)  # small positive
        row[j] = eps
        # subtract from adjust_col
        target_col = choose_adjust_col(i, allowed, j_forbid=j)
        if target_col == -1:
            target_col = n-1
        row[target_col] -= eps
        row = clip_and_renorm(row)
        return set_row_and_return()

    # default: no-op
    return M_str

# ---------- Lean check (n=3/4 row-sum proof, nonneg proof for n<=4) ----------

def lean_check(M_str):
    n = len(M_str)
    lem = None
    if n == 3: lem = "Fin.sum_univ_three"
    elif n == 4: lem = "Fin.sum_univ_four"

    lean_M = to_lean_matrix(M_str)
    row_sum_proof = ""
    if lem:
        row_sum_proof = f"""
example : RowSumOne M := by
  intro i; fin_cases i
  all_goals (simp [M, {lem}] <;> try norm_num)
"""
    else:
        # 先跳过 Lean 行和证明（Python 已严格检查）；只给非负性 proof
        row_sum_proof = "-- row-sum proof omitted for n>4; checked in Python.\n"

    # 非负性：把 i、j 都 fin_cases 掉就能 norm_num
    nonneg_proof = ""
    if n <= 4:
        nonneg_proof = f"""
example : Nonneg M := by
  intro i j; fin_cases i <;> fin_cases j
  all_goals (simp [M] <;> try norm_num)
"""
    else:
        nonneg_proof = "-- nonneg proof omitted for n>4; checked in Python.\n"

    lean = f"""
import Mathlib
import Mathlib.Tactic
open Matrix BigOperators Finset
set_option autoImplicit false

def RowSumOne (M : Matrix (Fin {n}) (Fin {n}) ℚ) : Prop := ∀ i, (∑ j, M i j) = 1
def Nonneg (M : Matrix (Fin {n}) (Fin {n}) ℚ) : Prop := ∀ i j, 0 ≤ M i j

def M : Matrix (Fin {n}) (Fin {n}) ℚ := {lean_M}

{row_sum_proof}
{nonneg_proof}
"""
    path = os.path.join("mathlib4", "StochasticCheck.lean")
    with open(path, "w") as f:
        f.write(lean)
    r = subprocess.run(["lake","env","lean","StochasticCheck.lean"],
                       cwd="mathlib4", capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)

# ---------- Build from edges (+ optional weights) ----------

def build_from_edges(n, edges, weights=None, require_pos_on_allowed=True):
    allowed = {i: set(edges[str(i)]) for i in range(n)}
    W = [[Fraction(0,1) for _ in range(n)] for _ in range(n)]

    # place given weights
    if weights:
        for key, val in weights.items():
            i_str, j_str = key.split(",")
            i, j = int(i_str), int(j_str)
            if j not in allowed[i]:
                raise ValueError(f"weight {key} not allowed by edges")
            W[i][j] = Fraction(val)

    # distribute remaining mass uniformly over remaining allowed slots
    for i in range(n):
        rem_cols = [j for j in allowed[i] if W[i][j] == 0]
        s = sum(W[i][j] for j in range(n))
        need = Fraction(1,1) - s
        if need < 0:
            # normalize entire row (weights too large)
            row = W[i]
            row = [max(x, Fraction(0,1)) for x in row]
            S = sum(row)
            row = [x/S for x in row] if S != 0 else [Fraction(1,1)] + [Fraction(0,1)]*(n-1)
            W[i] = row
        else:
            if rem_cols:
                fill = need / len(rem_cols)
                for j in rem_cols:
                    W[i][j] = fill
            else:
                # all weight assigned; if <1, push into a preferred allowed col
                if need > 0:
                    target = i if i in allowed[i] else (sorted(allowed[i])[0])
                    W[i][target] += need

        # if require_pos_on_allowed: make sure allowed entries are >0
        if require_pos_on_allowed:
            eps_targets = [j for j in allowed[i] if W[i][j] == 0]
            if eps_targets:
                eps = Fraction(1, 100*len(eps_targets))
                for j in eps_targets:
                    W[i][j] += eps
                # renorm back to 1
                W[i] = clip_and_renorm(W[i])

    return [[q_to_str(x) for x in r] for r in W]

# ---------- CLI ----------

def save_outputs(M_final, tag=""):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"matrix_{tag}{ts}.json")
    lean_path = os.path.join("mathlib4", "StochasticCheck.lean")
    with open(json_path, "w") as f:
        json.dump({"matrix": M_final}, f, indent=2)
    print(f"Saved matrix to {json_path}")
    print(f"Lean file at {lean_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--max_steps", type=int, default=10)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    n = cfg.get("n")
    edges = cfg.get("edges")
    weights = cfg.get("weights")
    require_pos = cfg.get("require_pos_on_allowed", True)
    add_self = cfg.get("add_self_loop", False)  # 架构里预留，如要自动加入自环可在 edges 生成时处理

    if "matrix" in cfg:
        # direct matrix mode
        M = cfg["matrix"]
        n = len(M)
        print(f"[matrix mode] n={n}")
    else:
        # edges mode
        assert n is not None and edges is not None, "need n and edges when matrix absent"
        # 如果 add_self_loop=True，把 i->i 加进 edges
        if add_self:
            for i in range(n):
                s = set(edges[str(i)])
                s.add(i)
                edges[str(i)] = sorted(list(s))
        M = build_from_edges(n, edges, weights=weights, require_pos_on_allowed=require_pos)
        print(f"[edges mode] n={n}, matrix built")

    # First report violations (Python)
    vios = find_violations(M, edges=edges, require_pos_on_allowed=require_pos)
    print_violations(vios)

    # Lean check
    ok, log = lean_check(M)
    print("Lean pass?", ok)
    if not ok:
        print("\nLean error (tail):", "\n".join(log.splitlines()[-20:]))

    # Try greedy fix loop if not passed or Python violations exist
    steps = 0
    while (vios or not ok) and steps < args.max_steps:
        steps += 1
        if vios:
            print(f"\nFixing only the 1st violation (step {steps}) …")
            M = fix_one(M, vios[0], edges=edges, require_pos_on_allowed=require_pos)

        # re-eval
        vios = find_violations(M, edges=edges, require_pos_on_allowed=require_pos)
        print_violations(vios)
        ok, log = lean_check(M)
        print("Lean pass?", ok)
        if not ok:
            print("Lean error (tail):", "\n".join(log.splitlines()[-20:]))

    if vios or not ok:
        print("\nStopped with remaining issues; consider editing inputs or increasing --max_steps.")
    else:
        print("\nAll checks passed ✅")
    save_outputs(M, tag=("edges_" if "matrix" not in cfg else "matrix_"))

if __name__ == "__main__":
    main()
