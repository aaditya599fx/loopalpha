# LoopAlpha research report

## Question

Can an optimizer use the individual contributions to a shared block's gradient to improve convergence at the same elapsed training time? A secondary experiment tests whether the optimizer rankings transfer to held-out stock-return ranking in simulation.

## Method

A flatten-and-project encoder maps feature windows to a hidden state. The network applies a single normalized residual MLP four times and predicts one scalar. For a tied parameter θ, the ordinary gradient is Σₖ ∇θₖL evaluated at equal-valued copies. The probe uses PyTorch's functional parameter substitution to create independent leaves for each occurrence and reads their gradients after one backward pass. The audit measures reconstruction error, gradient norms, pairwise alignment and instrumentation time over depths 1, 2, 4 and 8. [PyTorch functional call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html)

The experimental update feeds the mean of the occurrence gradients through AdamW. At fixed depth the mean differs from the ordinary tied gradient by a constant factor, which largely cancels under Adam normalization. The clipping and weighting variants can change the direction. The separate-moments variant averages normalized Adam directions after tracking moments for each occurrence. Baselines use momentum SGD, AdamW and Muon for eligible residual-block matrices. [PyTorch Muon](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)

## Findings

The gradients reconstructed the tied derivatives at all 12 seed/depth combinations. The largest absolute difference in float64 was 4.44 × 10⁻¹⁶. At depth four, median forward/backward time was approximately 0.49 ms for ordinary autograd and 0.64 ms for probing, with about 78 KiB of additional copy and gradient storage on the audit batch.

Muon achieved the lowest mean held-out MSE on the sequence task at the 120 ms cutoff (0.0396) and at 160 steps (0.0326). Momentum SGD reached the validation threshold in a median 33.9 ms. Equal-weight depth-aware AdamW matched ordinary AdamW at 160 steps to four decimals (0.0637), and its mean MSE was higher at 120 ms (0.0874 versus 0.0637). Separate moments lowered the fixed-step loss to 0.0592 but raised loss under the time budget to 0.1749. There were zero recorded failures across 291 tuning trials and 111 final seeded runs. The tested variants provide no evidence of improved time-budgeted convergence on this task.

Momentum SGD led the terminal simulated market fold by mean Rank IC (0.0907), with the linear model at 0.0894 and Muon at 0.0836. The best depth-aware variant by Rank IC used separate moments (0.0564). The deterministic momentum signal had the highest mean terminal net return under the 10 bps cost convention (4.65%) despite a lower Rank IC (0.0506). That difference illustrates the effect of tail selection and holding periods. The shuffled-label control averaged −0.0327 Rank IC and the noise control averaged −0.0062 in the terminal fold. Noise averaged 0.0544 in the earlier fold, and one of its three date-bootstrap intervals excluded zero. That outcome limits claims based on a single fold or seed.

The results support the gradient implementation and expose its compute cost. They leave the optimizer hypothesis unconfirmed under the selected tasks, budgets and tuning rule.

## Scope and limitations

- Both datasets are synthetic. The sequence generator defines a nonlinear recurrence and the panel generator defines three serial-dependence regimes. A result may reflect these specific tasks.
- The wall-clock endpoint uses the last completed validation checkpoint within budget and caps all trials at 160 updates. Checkpoint cadence, CPU load and run order affect measured times. The benchmark reports measured CPU wall time and analytically calculated probe tensor storage. It does not measure total peak process memory or GPU performance.
- Rates are selected using final validation MSE. This selection rule may favor a learning rate that performs better late in training. First-threshold times refer to successful threshold hits and are reported beside hit rates.
- The market panel is split by session. The second fold is the terminal holdout and contains 99 eligible decision sessions before any data-specific exclusions. Block bootstrap intervals describe date variation within a seed. Three training seeds offer limited information about training variability.
- Backtest execution assumes reported opening prices can be traded. Positions close in full at each cohort boundary. Transaction-cost stress covers explicit traded notional. Borrow, financing, market impact and capacity are outside the simulator.
- The historical-input interface still requires a licensed, documented panel. Price adjustments, point-in-time membership and delisting coverage must be assessed for the chosen feed.

## Reproduction

The README gives installation and run commands. The protocol fixes the evaluation budget, grid and controls. The generated results and raw JSON include outcomes for individual seeds, tuning, cost sensitivity and market uncertainty. The depth audit script regenerates the diagnostic JSON.
