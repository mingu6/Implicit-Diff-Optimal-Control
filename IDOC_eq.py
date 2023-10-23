import numpy as np


def itemsetter(items):
    def g(obj, values):
        for item, value in zip(items, values):
            obj[item] = value
    return g


def Traj_L2_Loss(demo, traj):
    state_traj = traj['state_traj_opt']
    control_traj = traj['control_traj_opt']
    demo_state_traj = demo['state_traj_opt']
    demo_control_traj = demo['control_traj_opt']

    dldx_traj = state_traj - demo_state_traj
    dldu_traj = control_traj - demo_control_traj
    loss = np.linalg.norm(dldx_traj) ** 2 + np.linalg.norm(dldu_traj) ** 2
    return loss


def build_blocks_idoc(auxsys_OC, delta=None):
    """
    This function takes the building blocks of the auxiliary COC system defined in the PDP paper
    and uses them to construct the blocks in our IDOC identities. 

    Inputs:

    auxsys_COC object: Dictionary with values being Jacobian/Hessian blocks of the constraints/cost.

    Outputs: - H_t: List of first T blocks in H
             - H_T: Final state block in H
             - A: All blocks in A (lower diagonal) corresponding to init. state, dynamics + eq. + ineq. (r_{-1}, r_{0}, ... r{T-1})
             - B_t: First T blocks of B
             - B_T: Final block of B
             - C: All blocks of C except C_0 (C_1, ..., C_T). C_0 is just zeros
             - ns: number of states
             - nc: number of controls
             - T: Horizon
    """

    T = len(auxsys_OC['dynF'])
    ns = auxsys_OC['Hxx'][0].shape[0]
    nc = auxsys_OC['Huu'][0].shape[0]
    nz = auxsys_OC['Hxe'][0].shape[1]

    # H blocks
    Hxx = np.stack(auxsys_OC['Hxx'], axis=0)
    Hxu = np.stack(auxsys_OC['Hxu'], axis=0)
    Huu = np.stack(auxsys_OC['Huu'], axis=0)
    H_t = np.block([[Hxx, Hxu], [Hxu.transpose(0, 2, 1), Huu]])
    H_T = auxsys_OC['hxx'][0]
    if delta is not None:
        H_t += delta * np.eye(ns+nc)[None, ...]
        H_T += delta * np.eye(ns)

    # A blocks
    dynFx = auxsys_OC['dynF']
    dynFu = auxsys_OC['dynG']
    
    A = -np.stack([np.block([dynFx[t], dynFu[t]])  for t in range(T)])

    # B blocks
    Hxe = np.stack(auxsys_OC['Hxe'], axis=0)
    Hue = np.stack(auxsys_OC['Hue'], axis=0)
    B_t = np.concatenate((Hxe, Hue), axis=1)
    B_T = auxsys_OC['hxe'][0]

    # C blocks
    dynE = auxsys_OC['dynE']
    C_0 = np.zeros((ns, nz))
    C = np.stack([C_0] + [-dynE[t] for t in range(T)])

    return H_t, H_T, A, B_t, B_T, C, ns, nc, T


def idoc_full(H, H_T_block, A_blocks, B, B_T_block, C, ns, nc, T):
    Hinv = np.linalg.inv(H)
    H_T_inv = np.linalg.inv(H_T_block)

    # compute and cache (H^1-A^T) expression in Prop. 4.5 in DDN paper
    HinvAT_diag_blocks = Hinv[..., :ns]
    HinvAT_diag_T_block = H_T_inv
    HinvAT_upper_blocks = Hinv @ A_blocks.transpose(0, 2, 1)

    # compute AH^-1B - C in Prop. 4.5 in DDN paper
    AHinvB_C = HinvAT_diag_blocks.transpose(0, 2, 1) @ B - C[:-1, ...]
    AHinvB_C[1:, ...] += HinvAT_upper_blocks[:-1, ...].transpose(0, 2, 1) @ B[:-1, ...]
    AHinvB_C_T_block = HinvAT_diag_T_block.T @ B_T_block - C[-1, ...]
    AHinvB_C_T_block += HinvAT_upper_blocks[-1, ...].T @ B[-1, ...]

    # compute AH^-1AT expression in Prop. 4.5 in DDN paper
    AHAinvAT_diag = np.concatenate((Hinv[..., :ns, :ns], H_T_inv[None, ...]), axis=0)
    AHAinvAT_diag[1:, ...] += A_blocks @ HinvAT_upper_blocks
    AHAinvAT_upper = HinvAT_upper_blocks[:, :ns, :].copy()
    AHAinvAT_lower = A_blocks @ HinvAT_diag_blocks

    # use Thomas's algorithm for block tridiagonal matrices to solve for (AH^-1A^T)^-1(AH^-1B - C)
    AHinvAT_AHinvB_C = [None] * T
    for t in range(1, T+1):
        CR = np.linalg.solve(AHAinvAT_diag[t-1], np.concatenate((AHAinvAT_upper[t-1], AHinvB_C[t-1]), axis=1))
        AHAinvAT_upper[t-1], AHinvB_C[t-1] = CR[:, :ns], CR[:, ns:]

        AHAinvAT_diag[t] -= AHAinvAT_lower[t-1] @ AHAinvAT_upper[t-1]
        if t == T:
            AHinvB_C_T_block = AHinvB_C_T_block - AHAinvAT_lower[t-1] @ AHinvB_C[t-1]
        else:
            AHinvB_C[t] -= AHAinvAT_lower[t-1] @ AHinvB_C[t-1]

    AHinvAT_AHinvB_C_T_block = np.linalg.solve(AHAinvAT_diag[T], AHinvB_C_T_block)
    AHinvAT_AHinvB_C[T-1] = AHinvB_C[T-1] - AHAinvAT_upper[T-1] @ AHinvAT_AHinvB_C_T_block
    for t in reversed(range(T-1)):  # backward recursion
        AHinvAT_AHinvB_C[t] =  AHinvB_C[t] - AHAinvAT_upper[t] @ AHinvAT_AHinvB_C[t+1]
    AHinvAT_AHinvB_C = np.stack(AHinvAT_AHinvB_C)

    # multiply by H^-1AT to get H^-1AT(AH^-1A^T)^-1(AH^-1B - C) (left term, projection to constraint surface)
    left_term = HinvAT_diag_blocks @ AHinvAT_AHinvB_C
    left_term[:-1] += HinvAT_upper_blocks[:-1] @ AHinvAT_AHinvB_C[1:]
    left_term[-1] += HinvAT_upper_blocks[-1] @ AHinvAT_AHinvB_C_T_block
    left_term_T_block = HinvAT_diag_T_block @ AHinvAT_AHinvB_C_T_block

    # solve right term (gradient of unconstrained problem) H^-1B and subtract from left term
    right_term = Hinv @ B
    right_term_T_block = H_T_inv @ B_T_block

    combined = [grad_ for grad_ in left_term - right_term]
    combined += [left_term_T_block - right_term_T_block]
    dxdp_traj_vec = [comb[:ns, :] for comb in combined]
    dudp_traj_vec = [comb[ns:, :] for comb in combined[:-1]]

    time_ = [k for k in range(T + 1)]
    sol_full = {'state_traj_opt': dxdp_traj_vec,
                'control_traj_opt': dudp_traj_vec,
                'time': time_}

    return sol_full


def idoc_vjp(demo_traj, traj, H, H_T_block, A_blocks, B, B_T_block, C, ns, nc, T):
    demo_state_traj = demo_traj['state_traj_opt']
    demo_control_traj = demo_traj['control_traj_opt']
    state_traj = traj['state_traj_opt']
    control_traj = traj['control_traj_opt']

    dldx_traj = state_traj - demo_state_traj
    dldu_traj = control_traj - demo_control_traj

    dldxu_traj = np.concatenate((dldx_traj[:-1, :], dldu_traj), axis=1)[:, None, :]
    dldx_T = dldx_traj[-1, :]

    Hinv = np.linalg.inv(H)
    H_T_inv = np.linalg.inv(H_T_block)

    # compute and cache (H^1-A^T) expression in Prop. 4.5 in DDN paper
    HinvAT_diag_blocks = Hinv[..., :ns]
    HinvAT_diag_T_block = H_T_inv
    HinvAT_upper_blocks = Hinv @ A_blocks.transpose(0, 2, 1)

    # compute AH^-1AT expression in Prop. 4.5 in DDN paper
    AHAinvAT_diag = np.concatenate((Hinv[..., :ns, :ns], H_T_inv[None, ...]), axis=0)
    AHAinvAT_diag[1:, ...] += A_blocks @ HinvAT_upper_blocks
    AHAinvAT_upper = HinvAT_upper_blocks[:, :ns, :].copy()
    AHAinvAT_lower = A_blocks @ HinvAT_diag_blocks

    # compute left VJP term, v^T H^-1AT
    left_term_blocks = dldxu_traj @ HinvAT_diag_blocks
    left_term_blocks[1:, ...] += dldxu_traj[:-1, ...] @ HinvAT_upper_blocks[:-1, ...]
    left_term_T_block = dldx_T @ HinvAT_diag_T_block + dldxu_traj[-1, ...] @ HinvAT_upper_blocks[-1, ...]

    # compute v^T H^-1AT (AH^-1AT)
    for t in range(1, T+1):
        CR = np.linalg.solve(AHAinvAT_diag[t-1].T, np.concatenate((AHAinvAT_lower[t-1].T, left_term_blocks[t-1].T), axis=1))
        AHAinvAT_lower[t-1], left_term_blocks[t-1] = CR[:, :ns].T, CR[:, ns:].T

        AHAinvAT_diag[t] -= AHAinvAT_lower[t-1] @ AHAinvAT_upper[t-1]
        update_block = left_term_T_block if t == T else left_term_blocks[t]
        update_block -= left_term_blocks[t-1] @ AHAinvAT_upper[t-1]

    left_term_T_block = (np.linalg.solve(AHAinvAT_diag[T], left_term_T_block.T)).T
    left_term_blocks[T-1] = left_term_blocks[T-1] - left_term_T_block @ AHAinvAT_lower[T-1]
    for t in reversed(range(T-1)):  # backward recursion
        left_term_blocks[t] =  left_term_blocks[t] - left_term_blocks[t+1] @ AHAinvAT_lower[t]
    left_term_blocks = np.stack(left_term_blocks)

    # compute v^T H^-1AT (AH^-1A^T)^-1 AH^-1
    left_term_blocks1 = left_term_blocks @ HinvAT_diag_blocks.transpose(0, 2, 1)
    left_term_blocks1[:-1, ...] += left_term_blocks[1:, ...] @ HinvAT_upper_blocks[:-1, ...].transpose(0, 2, 1)
    left_term_blocks1[-1, ...] += left_term_T_block @ HinvAT_upper_blocks[-1, ...].T
    left_term_T_block1 = left_term_T_block @ HinvAT_diag_T_block.T
    
    # compute next step left to right VJP v^T H^-1AT (AH^-1A^T)^-1 AH^-1B
    left_term_blocks1 = left_term_blocks1 @ B
    left_term_T_block1 = left_term_T_block1 @ B_T_block

    # compute next step left to right VJP v^T H^-1AT (AH^-1A^T)^-1 (- C)
    left_term_blocks = left_term_blocks1 - left_term_blocks @ C[:-1, ...]
    left_term_T_block = left_term_T_block1 - left_term_T_block @ C[-1, ...]
    left_term = left_term_blocks.sum(axis=0) + left_term_T_block

    # right term VJP
    right_term = (dldxu_traj @ Hinv @ B).sum(axis=0)
    right_term += (dldx_T @ H_T_inv @ B_T_block)[None, :]

    dxudp = left_term - right_term
    return dxudp
