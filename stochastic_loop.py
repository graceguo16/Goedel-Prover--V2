# -*- coding: utf-8 -*-
import subprocess, os
from fractions import Fraction
# ---------- LLN: 模拟演示（Bernoulli 例子） ----------
import random
from statistics import mean

def simulate_lln_bernoulli(p=0.3, n_trials=20000, report_at=(100, 200, 500, 1000, 2000, 5000, 10000, 20000)):
    """
    模拟伯努利(p) 的大数定律：逐步计算样本均值 X̄_n，看它如何逼近 p。
    仅用文本报告（无图），便于在服务器/终端跑。
    """
    s = 0
    report_at = sorted(set([k for k in report_at if 1 <= k <= n_trials]))
    idx = 0
    last_mean = None
    print("\n[LLN simulation] Bernoulli(p={:.3f}), trials={}".format(p, n_trials))
    for t in range(1, n_trials + 1):
        x = 1 if random.random() < p else 0
        s += x
        if idx < len(report_at) and t == report_at[idx]:
            m = s / t
            print("  n={:<6d}  sample mean={:.6f}   |mean - p|={:.6f}".format(t, m, abs(m - p)))
            last_mean = m
            idx += 1
    if last_mean is None:
        last_mean = s / n_trials
    print("  final: n={}, sample mean={:.6f},  |mean - p|={:.6f}".format(n_trials, last_mean, abs(last_mean - p)))

def lln_tail_prob_demo(p=0.3, n_trials=20000, eps_list=(0.02, 0.05, 0.1)):
    """
    近似展示 P(|X̄_n - p| > eps) 随 n 变小：对每个 eps，取后半段窗口统计“越界比例”作为经验近似。
    这不是形式证明，只是直观感受“越界概率→0”的趋势。
    """
    s = 0
    means = []
    for t in range(1, n_trials + 1):
        x = 1 if random.random() < p else 0
        s += x
        means.append(s / t)
    half = n_trials // 2
    window = means[half:]  # 用后半段作为“n 足够大”的近似区间
    print("\n[LLN tail-prob (empirical)] using last {} steps".format(len(window)))
    for eps in eps_list:
        tail = sum(1 for m in window if abs(m - p) > eps) / len(window)
        print("  approx P(|X̄_n - p| > {:>.3f})  ≈ {:.6f}".format(eps, tail))
st.

# ---------- Lean 验证 ----------
def to_lean_matrix_3x3(mat):
    rows = [", ".join(r) for r in mat]
    return "!![ " + " ;\n      ".join(rows) + " ]"

def lean_check(mat):
    lean_M = to_lean_matrix_3x3(mat)
    code = f"""
import Mathlib
import Mathlib.Tactic
open Matrix BigOperators Finset
set_option autoImplicit false
def RowSumOne3 (M : Matrix (Fin 3) (Fin 3) ℚ) : Prop := ∀ i, (∑ j, M i j) = 1
def M : Matrix (Fin 3) (Fin 3) ℚ := {lean_M}
example : RowSumOne3 M := by
  intro i; fin_cases i
  all_goals (simp [M, Fin.sum_univ_three] <;> try norm_num)
"""
    path = os.path.join("mathlib4","StochasticCheck.lean")
    with open(path,"w") as f: f.write(code)
    r = subprocess.run(["lake","env","lean","StochasticCheck.lean"],
                       cwd="mathlib4", capture_output=True, text=True)
    return r.returncode == 0, (r.stdout+r.stderr)

# 备选：一次性“全修最后一列”
def renorm_last_entry(mat):
    fixed=[]
    for row in mat:
        fr = [Fraction(x) for x in row]
        target = Fraction(1,1) - sum(fr[:-1])
        if target < 0: target = Fraction(0,1)
        fixed.append([str(x) for x in fr[:-1]] + [f"{target.numerator}/{target.denominator}"])
    return fixed

# ---------- subgoals check ----------
def to_frac_matrix(M):
    return [[Fraction(x) for x in row] for row in M]

def find_markov_violations(M):
    F = to_frac_matrix(M)
    vios = []
    for i, row in enumerate(F):
        for j, v in enumerate(row):
            if v < 0:
                vios.append({"kind":"neg", "row":i, "col":j, "value":v})
        s = sum(row)
        if s != Fraction(1,1):
            vios.append({"kind":"row_sum", "row":i, "sum":s, "expected":Fraction(1,1)})
    return vios

def print_violations(vios):
    if not vios:
        print("No violations. ✅")
        return
    print(f"Violations ({len(vios)}):")
    for k, v in enumerate(vios, 1):
        if v["kind"] == "neg":
            i, j, val = v["row"], v["col"], v["value"]
            print(f"  {k}) NEGATIVE entry at row {i+1}, col {j+1}: {val}")
        else:
            i, s, exp = v["row"], v["sum"], v["expected"]
            print(f"  {k}) ROW-SUM ≠ 1 at row {i+1}: sum={s} (expected {exp})")

def fix_one_violation(M, vio):
    F = to_frac_matrix(M)
    i = vio["row"]
    row = F[i].copy()

    def set_last_entry_to_keep_sum(row_):
        s = sum(row_[:-1])
        row_[-1] = Fraction(1,1) - s
        return row_

    if vio["kind"] == "neg":
        j = vio["col"]
        bump = -row[j]
        row[j] = Fraction(0,1)
        row[-1] = row[-1] - bump
        row = set_last_entry_to_keep_sum(row)
    else:  # row_sum
        row = set_last_entry_to_keep_sum(row)

    
    if any(x < 0 for x in row) or sum(row) != Fraction(1,1):
        row = [max(x, Fraction(0,1)) for x in row]
        s = sum(row)
        row = [x/s for x in row] if s != 0 else row

    F[i] = row
    def frac_to_str(q): return f"{q.numerator}/{q.denominator}"
    return [[frac_to_str(x) for x in r] for r in F]

def fix_selected(M, vios, selected_ids):
    """
    selected_ids: 1-based 编号列表，例如 [2] 或 [1,3]
    逐个按序修；每修一次不重算 vios（保持你点名的 subgoals 语义稳定）
    """
    Mcur = [row[:] for row in M]
    for sid in selected_ids:
        idx = sid - 1
        if 0 <= idx < len(vios):
            Mcur = fix_one_violation(Mcur, vios[idx])
    return Mcur

def pretty(mat):
    for r in mat:
        print("  [" + ", ".join(r) + "]")

def ask_user_selection(vios):
    """
      - 输入 all / a / 空行：修全部
      - 输入 none / n：不修
      - 输入 逗号分隔编号：只修这些
    返回 ('all' | 'none' | [ids])
    """
    if not vios:
        return 'none'
    print("\nyou have errors above, how do you want me to fix？")
    print("  - all/a：fix all mistakes automatically")
    print("  - none/n：manually fix without using algorithm")
    print("  - type row numbers：fix mentioned-rows only ")
    s = input("> ").strip().lower()
    if s in ("", "all", "a"):
        return 'all'
    if s in ("none", "n"):
        return 'none'
    try:
        ids = [int(x) for x in s.replace("，",",").split(",") if x.strip()]
        return ids
    except:
        print("type illegal，none by default")
        return 'none'

# ---------- main ----------
if __name__ == "__main__":
    print("Choose mode:")
    print("  1) matrix_check  —— 继续做随机矩阵（行和=1、非负）检测/修补（你现在已有的功能）")
    print("  2) lln_demo      —— 演示大数定律（Bernoulli 仿真：样本均值→期望）")
    mode = input("> ").strip()

    if mode in ("2", "lln_demo", "lln"):
        # --- LLN 演示 ---
        try:
            p = float(input("Bernoulli p (default 0.3): ").strip() or "0.3")
            n = int(input("trials n (default 20000): ").strip() or "20000")
        except:
            p, n = 0.3, 20000
        simulate_lln_bernoulli(p=p, n_trials=n)
        lln_tail_prob_demo(p=p, n_trials=n)
else:
    
    M0 = [["1/3","1/12","1/10"],["0","1","1"],["1/3","1/3","1/3"]]

    ok, log = lean_check(M0)
    print("Pass?", ok)
    if not ok:
        vios = find_markov_violations(M0)
        print_violations(vios)

        selection = ask_user_selection(vios)
        if selection == 'none':
            print("\ndon't fix")
            exit(0)
        elif selection == 'all':
            print("\nFixing ALL subgoals …")
            M1 = renorm_last_entry(M0)  # 或者：逐个 fix_selected(M0, vios, list(range(1,len(vios)+1)))
        else:
            print(f"\nFixing subgoals {selection} …")
            M1 = fix_selected(M0, vios, selection)

        print("\nMatrix after fix:")
        pretty(M1)

        ok2, log2 = lean_check(M1)
        print("\nPass after fix?", ok2)
        if not ok2:
            
            print("Lean error (tail):", "\n".join(log2.splitlines()[-20:]))
