# Implicit Differentiation for Optimal Control (IDOC)
This repository contains a reference implementation for the paper "Revisiting Implicit Differentiation for Learning Problems in Optimal Control" (NeurIPS 2023) for the settings with and without inequality constraints in the forward optimal control problem. 

Our method, named "Implicit Differentiation for Optimal Control" (IDOC) is used to __differentiate through optimal control problems__ (sensitivity analysis), and improves on previous methods such as [DiffMPC](https://github.com/locuslab/differentiable-mpc), [PDP](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming) and [Safe-PDP](https://github.com/wanxinjin/Safe-PDP). IDOC can handle (smooth) optimal control problems which are

* Non-convex (non-linear dynamics and non-convex cost and constraint functions)
* Inequality constrained
* Contains non-linear equality constraints in addition to dynamics (such as terminal constraints)

For more details, see our [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/bcfcf7232cb74e1ef82d751880ff835b-Paper-Conference.pdf). Our code integrates seamlessly with the Safe-PDP codebase.

## Trajectory Derivatives

We provide the implementation of our trajectory derivative computations within the `IDOC_eq.py` (no inequality constraints) and `IDOC_ineq.py` (inequality constraints) file. 

## Integration with Safe-PDP

We provide an example training script which can be used with the Safe-PDP [codebase](https://github.com/wanxinjin/Safe-PDP/tree/main) for the imitation learning task on the cartpole environment where inequality constraints are present (`CIOC_Cartpole_IDOC.py`). Follow these steps to get started:

* Clone the Safe-PDP [repository](https://github.com/wanxinjin/Safe-PDP/tree/main)
* Install all dependencies required to run Safe-PDP (e.g., CasADi)
* Place the `IDOC_*.py` files into the `Safe-PDP/` folder found in the root directory of the Safe-PDP project. 
* Place the `CIOC_Carpole_IDOC.py` file into the `Examples/MPC/CIOC/` folder. 
* Run `CIOC_Carpole_IDOC.py`, which will solve the CIOC problem using IDOC trajectory derivatives!
