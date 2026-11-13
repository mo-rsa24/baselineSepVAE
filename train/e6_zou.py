"""
E6 — Joint Disentanglement (Zou et al. 2022)
=============================================
Key new ingredients vs our codebase:
  1. Label-conditioned prior p(z_s|y) = N(μ_y, I)  [LabelConditionedPrior]
  2. Classification loss CE(p(y|z_s), y)
  3. MI penalty MI(z_c, z_s)

Loss:
    MSE + KL_c + KL(z_s || p(z_s|y)) + γ·CE(clf(z_s), y) + δ·MI(z_c, z_s)

TODO: Implement full training loop using losses/conditional_prior.py.
      Stub only.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    print("[E6] Zou Joint Disentanglement — not yet implemented.")
    print("     Will use losses/conditional_prior.py:LabelConditionedPrior")
    print("     and losses/conditional_prior.py:kl_to_label_prior")

if __name__ == '__main__':
    main()
