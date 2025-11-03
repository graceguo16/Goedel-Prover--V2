import Mathlib
import Mathlib.Tactic
open Matrix BigOperators Finset
set_option autoImplicit false
def RowSumOne3 (M : Matrix (Fin 3) (Fin 3) ℚ) : Prop := ∀ i, (∑ j, M i j) = 1
def M : Matrix (Fin 3) (Fin 3) ℚ := !![ 1/3, 1/12, 7/12 ;
      0, 1, 0/1 ;
      1/3, 1/3, 1/3 ]
example : RowSumOne3 M := by
  intro i; fin_cases i
  all_goals (simp [M, Fin.sum_univ_three] <;> try norm_num)
