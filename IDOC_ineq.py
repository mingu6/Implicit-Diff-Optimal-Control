import numpy as np

def itemsetter(items):
    def g(obj, values):
        for item, value in zip(items, values):
            obj[item] = value
    return g


def build_blocks_idoc(auxsys_COC, delta=None):
    """
    This function takes the building blocks of the auxiliary COC system defined in the Safe-PDP paper
    and uses them to construct the blocks in our IDOC identities. 

    Inputs:

    auxsys_COC object: Dictionary with values being Jacobian/Hessian blocks of the constraints/cost.

    Outputs: - H_t: List of first T blocks in H
             - H_T: Final state block in H
             - A_t: First T+1 blocks in A (lower diagonal) corresponding to init. state, dynamics + eq. + ineq. (r_{-1}, r_{0}, ... r{T-1})
             - A_T: Final equality + inequality constraints block (no dynamics) (r_T)
             - B_t: First T blocks of B
             - B_T: Final block of B
             - C_t: First T blocks of C except C_0 (C_1, ..., C_T). C_0 is just zeros
             - C_T: Final block of C
             - ns: number of states
             - nc: number of controls
             - T: Horizon

    """
    T = auxsys_COC['horizon']
    ns = auxsys_COC['Lxx_t'][0].shape[0]
    nc = auxsys_COC['Luu_t'][0].shape[0]

    # H blocks
    Lxx_t = np.stack(auxsys_COC['Lxx_t'], axis=0)
    Lxu_t = np.stack(auxsys_COC['Lxu_t'], axis=0)
    Luu_t = np.stack(auxsys_COC['Luu_t'], axis=0)
    H_t = np.block([[Lxx_t, Lxu_t], [Lxu_t.transpose(0, 2, 1), Luu_t]])
    H_T = auxsys_COC['Lxx_T'][0]
    if delta is not None:
        H_t += delta * np.eye(ns+nc)[None, ...]
        H_T += delta * np.eye(ns)

    # A blocks
    GbarHx_t = auxsys_COC['GbarHx_t']
    GbarHu_t = auxsys_COC['GbarHu_t']
    GbarHx_T = auxsys_COC['GbarHx_T'][0]
    dynFx_t = auxsys_COC['dynFx_t']
    dynFu_t = auxsys_COC['dynFu_t']
    
    A_t = [np.block([[GbarHx_t[t], GbarHu_t[t]], [dynFx_t[t], dynFu_t[t]]])  for t in range(T)]
    A_T = GbarHx_T

    # B blocks
    Lxe_t = np.stack(auxsys_COC['Lxe_t'], axis=0)
    Lue_t = np.stack(auxsys_COC['Lue_t'], axis=0)
    B_t = np.concatenate((Lxe_t, Lue_t), axis=1)
    B_T = auxsys_COC['Lxe_T'][0]

    # C blocks
    GbarHe_t = auxsys_COC['GbarHe_t']
    dynFe_t = auxsys_COC['dynFe_t']
    C_t = [np.concatenate((GbarHe_t[t], dynFe_t[t]), axis=0) for t in range(T)]
    C_T = auxsys_COC['GbarHe_T'][0]

    return H_t, H_T, A_t, A_T, B_t, B_T, C_t, C_T, ns, nc, T


def idoc_full(H_t, H_T, A_t, A_T, B_t, B_T, C_t, C_T, ns, nc, T):
    nz = B_T.shape[1]
    
    Hinv_t = np.linalg.inv(H_t)
    Hinv_T = np.linalg.inv(H_T)

    A_t_sz = np.array([A_.shape[0] for A_ in A_t])  # first block treated separately for AH^-1AT
    A_sz_uniq = set(A_t_sz)

    # compute and cache (H^1-A^T) expression in Prop. 4.6 in DDN paper
    HinvAT_upper_t_blocks = T * [None]
    HinvAT_upper_T_block = Hinv_T @ A_T.T
    HinvAT_lower_t_blocks = -Hinv_t[..., :ns]
    HinvAT_lower_t_blocks[0, ...] *= -1.
    HinvAT_lower_T_block = -Hinv_T
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        A_vec = np.stack([A_t[ind] for ind in inds], axis=0)
        itemsetter(inds)(HinvAT_upper_t_blocks, Hinv_t[inds] @ A_vec.transpose(0, 2, 1))

    # compute AH^-1B - C in Prop. 4.5 in DDN paper
    AHinvB_C = (T + 2) * [None]
    AHinvB_C[0] = Hinv_t[0][:, :ns].T @ B_t[0]
    AHinvB_C[-1] = HinvAT_upper_T_block.T @ B_T - C_T
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        HinvAT_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        C_vec = np.stack([C_t[ind] for ind in inds], axis=0)
        lhs_vec = HinvAT_vec.transpose(0, 2, 1) @ B_t[inds, ...] - C_vec
        if T-1 not in inds:
            lhs_vec[:, -ns:, :] -= Hinv_t[inds+1, :, :ns].transpose(0, 2, 1) @ B_t[inds+1, ...]
        else:
            lhs_vec[:-1, -ns:, :] -= Hinv_t[inds[:-1]+1, :, :ns].transpose(0, 2, 1) @ B_t[inds[:-1]+1, ...]
            lhs_vec[-1, -ns:, :] -= Hinv_T[:, :ns].T @ B_T
        itemsetter(inds+1)(AHinvB_C, lhs_vec)           

    # compute AH^-1AT expression in Prop. 4.6 in DDN paper
    diag_blk = (T + 2) * [None]
    upper_blk = (T + 1) * [None]
    # set diag end blocks
    diag_blk[0] = Hinv_t[0, :ns, :ns].copy()
    diag_blk[-1] = A_T @ HinvAT_upper_T_block
    # set upper -1
    upper_blk[-1] = -HinvAT_upper_T_block[:ns, :].copy()
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        A_vec = np.stack([A_t[ind] for ind in inds], axis=0)
        HinvAT_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        # diagonal blocks
        diag_blocks = A_vec @ HinvAT_vec
        if T-1 not in inds:
            diag_blocks[:, -ns:, -ns:] += Hinv_t[inds+1, :ns, :ns]
        else:
            diag_blocks[:-1, -ns:, -ns:] += Hinv_t[inds[:-1]+1, :ns, :ns]
            diag_blocks[-1, -ns:, -ns:] += Hinv_T
        itemsetter(inds+1)(diag_blk, diag_blocks)
        # upper blocks
        upper = -HinvAT_vec[:, :ns, :].copy()
        if 0 in inds:
            upper[0, ...] *= -1.
        itemsetter(inds)(upper_blk, upper)

    lower_blk = [upper_blk[0].T.copy()]
    for t in range(1, T):
        sz = A_t_sz[t-1]
        upper_blk[t] = np.concatenate((np.zeros((sz-ns, upper_blk[t].shape[1])), upper_blk[t]), axis=0)
        lower_blk.append(upper_blk[t].T.copy())
    upper_blk[T] = np.concatenate((np.zeros((A_t_sz[-1]-ns, upper_blk[T].shape[1])), upper_blk[T]), axis=0)
    lower_blk.append(upper_blk[T].T.copy())

    # use Thomas's algorithm for block tridiagonal matrices to solve for (AH^-1A^T)^-1(AH^-1B - C)
    AHinvAT_AHinvB_C = [None] * (T+2)
    for t in range(1, T+2):
        CR = np.linalg.solve(diag_blk[t-1], np.concatenate((upper_blk[t-1], AHinvB_C[t-1]), axis=1))
        n_lhs = upper_blk[t-1].shape[1]
        upper_blk[t-1], AHinvB_C[t-1] = CR[:, :n_lhs], CR[:, n_lhs:]

        diag_blk[t] -= lower_blk[t-1] @ upper_blk[t-1]
        AHinvB_C[t] -= lower_blk[t-1] @ AHinvB_C[t-1]

    AHinvAT_AHinvB_C[-1] = np.linalg.solve(diag_blk[T+1], AHinvB_C[-1])
    AHinvAT_AHinvB_C[T] = AHinvB_C[T] - upper_blk[T] @ AHinvAT_AHinvB_C[-1]
    for t in reversed(range(T)):  # backward recursion
        AHinvAT_AHinvB_C[t] = AHinvB_C[t] - upper_blk[t] @ AHinvAT_AHinvB_C[t+1]

    # multiply by H^-1AT to get H^-1AT(AH^-1A^T)^-1(AH^-1B - C) (left term, projection to constraint surface)
    rhs_bottom_vec = np.stack([blk[-ns:, :] for blk in AHinvAT_AHinvB_C[:-2]], axis=0)
    left_term = HinvAT_lower_t_blocks @ rhs_bottom_vec
    left_term_T = HinvAT_lower_T_block @ AHinvAT_AHinvB_C[-2] + HinvAT_upper_T_block @ AHinvAT_AHinvB_C[-1]
    left_term = np.concatenate((left_term.reshape(-1, nz), left_term_T), axis=0)

    left_term1 = T * [None]
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        rhs_vec = np.stack([AHinvAT_AHinvB_C[ind+1] for ind in inds], axis=0)
        HinvAT_block_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        left_block = HinvAT_block_vec @ rhs_vec
        itemsetter(inds)(left_term1, left_block)
    left_term[:-ns, :] += np.concatenate(left_term1, axis=0)

    # solve right term (gradient of unconstrained problem) H^-1B and subtract from left term
    left_term[:-ns, :] -= (Hinv_t @ B_t).reshape(-1, nz)
    left_term[-ns:, :] -= Hinv_T @ B_T 

    # unpack full
    dxu_dp_t = left_term[:-ns, :].reshape(T, ns+nc, nz)
    dx_dp = np.concatenate((dxu_dp_t[:, :ns, :], left_term[None, -ns:, :]), axis=0)
    du_dp = dxu_dp_t[:, ns:, :]

    time_ = [k for k in range(T + 1)]
    sol_full = {'state_traj_opt': dx_dp,
                'control_traj_opt': du_dp,
                'time': time_}

    return sol_full


def idoc_vjp(demo_traj, traj, H_t, H_T, A_t, A_T, B_t, B_T, C_t, C_T, ns, nc, T):
    nz = B_T.shape[1]

    demo_state_traj = demo_traj['state_traj_opt']
    demo_control_traj = demo_traj['control_traj_opt']
    state_traj = traj['state_traj_opt']
    control_traj = traj['control_traj_opt']

    dldx_traj = state_traj - demo_state_traj
    dldu_traj = control_traj - demo_control_traj

    dldxi_t_blocks = np.concatenate((dldx_traj[:-1], dldu_traj), axis=1)
    dldxi_T = dldx_traj[-1]
    Hinv_t = np.linalg.inv(H_t)
    Hinv_T = np.linalg.inv(H_T)

    A_t_sz = np.array([A_.shape[0] for A_ in A_t])  # first block treated separately for AH^-1AT
    A_sz_uniq = set(A_t_sz)

    # compute and cache (H^1-A^T) expression in Prop. 4.6 in DDN paper
    HinvAT_upper_t_blocks = T * [None]
    HinvAT_upper_T_block = Hinv_T @ A_T.T
    HinvAT_lower_t_blocks = -Hinv_t[..., :ns]
    HinvAT_lower_t_blocks[0, ...] *= -1.
    HinvAT_lower_T_block = -Hinv_T
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        A_vec = np.stack([A_t[ind] for ind in inds], axis=0)
        itemsetter(inds)(HinvAT_upper_t_blocks, Hinv_t[inds] @ A_vec.transpose(0, 2, 1))

    # compute AH^-1AT expression in Prop. 4.6 in DDN paper
    diag_blk = (T + 2) * [None]
    upper_blk = (T + 1) * [None]
    # set diag end blocks
    diag_blk[0] = Hinv_t[0, :ns, :ns].copy()
    diag_blk[-1] = A_T @ HinvAT_upper_T_block
    # set upper -1
    upper_blk[-1] = -HinvAT_upper_T_block[:ns, :].copy()
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        A_vec = np.stack([A_t[ind] for ind in inds], axis=0)
        HinvAT_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        # diagonal blocks
        diag_blocks = A_vec @ HinvAT_vec
        if T-1 not in inds:
            diag_blocks[:, -ns:, -ns:] += Hinv_t[inds+1, :ns, :ns]
        else:
            diag_blocks[:-1, -ns:, -ns:] += Hinv_t[inds[:-1]+1, :ns, :ns]
            diag_blocks[-1, -ns:, -ns:] += Hinv_T
        itemsetter(inds+1)(diag_blk, diag_blocks.transpose(0, 2, 1))
        # upper blocks
        upper = -HinvAT_vec[:, :ns, :].copy()
        if 0 in inds:
            upper[0, ...] *= -1.
        itemsetter(inds)(upper_blk, upper)

    lower_blk = [upper_blk[0].T.copy()]
    for t in range(1, T):
        sz = A_t_sz[t-1]
        upper_blk[t] = np.concatenate((np.zeros((sz-ns, upper_blk[t].shape[1])), upper_blk[t]), axis=0)
        lower_blk.append(upper_blk[t].T.copy())
    upper_blk[T] = np.concatenate((np.zeros((A_t_sz[-1]-ns, upper_blk[T].shape[1])), upper_blk[T]), axis=0)
    lower_blk.append(upper_blk[T].T.copy())

    # compute left VJP term, v^T H^-1AT
    left_term_blocks1 = dldxi_t_blocks[:, None, :] @ HinvAT_lower_t_blocks
    left_term_blocks = (T + 2) * [None]
    left_term_blocks[0] = left_term_blocks1[0, ...]
    left_term_blocks[-2] = (dldxi_T @ HinvAT_lower_T_block + dldxi_t_blocks[-1] @ HinvAT_upper_t_blocks[-1])[None, :]
    left_term_blocks[-1] = (dldxi_T @ HinvAT_upper_T_block)[None, :]
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz[:-1] == sz)[0]
        HinvAT_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        blks = dldxi_t_blocks[inds, None, :] @ HinvAT_vec
        blks[..., -ns:] += left_term_blocks1[inds+1, :]
        itemsetter(inds+1)(left_term_blocks, blks)

    # use Thomas's algorithm for block tridiagonal matrices to solve for (AH^-1A^T)^-1(AH^-1B - C)
    for t in range(1, T+2):
        CR = np.linalg.solve(diag_blk[t-1], np.concatenate((upper_blk[t-1], left_term_blocks[t-1].T), axis=1))
        n_lhs = upper_blk[t-1].shape[1]
        upper_blk[t-1], left_term_blocks[t-1] = CR[:, :n_lhs], CR[:, n_lhs:].T

        diag_blk[t] -= lower_blk[t-1] @ upper_blk[t-1]
        left_term_blocks[t] -= (lower_blk[t-1] @ left_term_blocks[t-1].T).T

    left_term_blocks[-1] = np.linalg.solve(diag_blk[T+1], left_term_blocks[-1].T).T
    left_term_blocks[T] = left_term_blocks[T] - (upper_blk[T] @ left_term_blocks[-1].T).T
    for t in reversed(range(T)):  # backward recursion
        left_term_blocks[t] = left_term_blocks[t] - (upper_blk[t] @ left_term_blocks[t+1].T).T

    # compute v^T H^-1AT (AH^-1A^T)^-1 AH^-1
    left_last_vec = np.stack([blk[:, -ns:] for blk in left_term_blocks[:T]], axis=0)
    left_term_blocks2 = left_last_vec @ HinvAT_lower_t_blocks.transpose(0, 2, 1)
    left_term_blocks1 = (T + 1) * [None]
    left_term_blocks1[-1] = (left_term_blocks[-1] @ HinvAT_upper_T_block.T + left_term_blocks[-2] @ HinvAT_lower_T_block.T).squeeze(0)
    for sz in A_sz_uniq:
        inds = np.where(A_t_sz == sz)[0]
        HinvAT_vec = np.stack([HinvAT_upper_t_blocks[ind] for ind in inds], axis=0)
        left_vec = np.stack([left_term_blocks[ind+1] for ind in inds], axis=0)
        blks = (left_vec @ HinvAT_vec.transpose(0, 2, 1) + left_term_blocks2[inds, ...]).squeeze(1)
        itemsetter(inds)(left_term_blocks1, blks)

    # compute v^T H^-1AT (AH^-1A^T)^-1 (AH^-1B - C)
    dp = np.concatenate(left_term_blocks1) @ np.concatenate((B_t.reshape(-1, nz), B_T), axis=0)
    dp -= np.concatenate([blk.squeeze(0) for blk in left_term_blocks]) @ np.concatenate([np.zeros((ns, nz))] + C_t + [C_T], axis=0)
    
    # right term VJP
    dp -= (dldxi_t_blocks[:, None, :] @ Hinv_t @ B_t).sum(axis=(0,1)) + dldxi_T @ Hinv_T @ B_T

    return dp
