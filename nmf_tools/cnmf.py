import numpy as np
from .valid_param import ValidateParameters

eps = 0.0001

def cmfwisa(V, num_basis_elems, config = None):
    if not config:
        config = {}
    m, n = V.shape()
    if isinstance(num_basis_elems, list):
        num_basis_elems = [num_basis_elems]
    num_sources = len(num_basis_elems)

    [config, is_W_cell, is_H_cell] = ValidateParameters(config, V, num_basis_elems)
    is_P_cell = False
    # 创建初试相位矩阵 Initialize phase matrices
    if 'P_init' not in config.keys() or not config["P_init"]:  # not given any inital phase matrices. Fill these in.
        if num_sources == 1:
            is_P_cell = False
        else:
            is_P_cell = True

        config.P_init = [] # cell(num_sources, 1);
        for i in range(num_sources):# = 1 : num_sources
            config.P_init.append(np.exp(1j * np.angle(V)))

    elif isinstance(config["P_init"], list) and len(config["P_init"]) != num_sources:  # given an incorrect number of initial phase matrices
        raise Exception('Requested %d encoding matrices. Given %d initial phase matrices.' % (num_sources, len(config.P_init)))
    elif not isinstance(config["P_init"], list): # given a matrix
        is_P_cell = False
        config["P_init"] = [config["P_init"], ]
    else:  # organize phase matrices as {P_1; P_2; ...; P_num_sources}
        is_P_cell = True
        config["P_init"] = config["P_init"][:]

    # Update switches for phase matrices
    if 'P_fixed' not in config.keys() or not config["P_fixed"]: # not given an update switch. Fill this in.
        config["P_fixed"] = []
        for i in range(num_sources):
            config["P_fixed"].append(False)
    elif isinstance(config["P_fixed"], list) and len(config["P_fixed"]) > 1 and len(config["P_fixed"]) != num_sources: # given an incorrect number of update switches
        raise Exception("Requested %d basis matrices. Given %d update switches." % (num_sources, len(config.P_fixed)))
    elif not isinstance(config["P_fixed"]) or len(config["P_fixed"]) == 1: # extend one update switch level to all phase matrices
        if isinstance(config["P_fixed"], list):
            temp = config["P_fixed"][0]
        else:
            temp = config["P_fixed"]
        config.P_fixed = []
        for i in range(num_sources):
            config["P_fixed"].append(np.copy(temp))

    W = config["W_init"]
    for i in range(num_sources):
        W[i] = W[i] * np.diag(1 / np.sqrt(np.sum(W[i]**2, 1)))

    H = config["H_init"]

    W_all = np.vstack(W)
    H_all = np.vstack(H)

    P = config["P_init"]
    beta = [np.zeros([m, n]) for i in range(num_sources)]
    V_hat_per_source = []
    for i in range(num_sources):
        V_hat_per_source.append(np.dot(W[i], H[i]) * P[i])

    V_hat = np.sum(V_hat_per_source, axis=3)

    V_bar_per_source = [np.zeros([m, n]) for _ in range(num_sources)]

    cost = np.zeros(config["maxiter"])

    for iter in range(config["maxiter"]):
        # Update auxiliary variables
        for i in range(num_sources):
            beta[i] = np.dot(W[i], H[i]) / np.dot(W_all, H_all)
            V_bar_per_source[i] = V_hat_per_source[i] + beta[i] * (V - V_hat)

        # Update phase matrices
        for i in range(num_sources):
            if "P_fixed" not in config.keys() or not config["P_fixed"][i]:
                P[i] = np.exp(1j * np.angle(V_bar_per_source[i])) # V_bar_per_source(:, :, i) ./ abs(V_bar_per_source(:, :, i));

        # Update basis matrices
        for i in range(num_sources):
            if "W_fixed" not in config.keys() or config["W_fixed"][i]:
                W[i] = W[i] * (((np.abs(V_bar_per_source[i]) / beta[i]) * H[i].H) / np.max(np.dot(np.dot(W_all, H_all), H[i].H), eps))
                W[i] = W[i] * np.diag(1 / np.sqrt(np.sum(W[i]**2, 1)))

        # Update encoding matrices
        for i in range(num_sources):
            if "H_fixed" not in config.keys() or not config["H_fixed"][i]:
                H[i] = H[i] * ((W[i].H * (np.abs(V_bar_per_source[i]) / beta[i])) / np.max(np.dot(np.dot(W[i].H, W_all), H_all) + config["H_sparsity"][i], eps)) # max(H .* ((W.^2).H * (ones(m, n) ./ beta)) + config.H_sparsity, eps));


        W_all = np.vstack(W)
        H_all = np.vstack(H)

        for i in range(num_sources):
            V_hat_per_source[i] = np.dot(W[i], H[i]) * P[i]

        V_hat = np.sum(V_hat_per_source, axis=3)

        # Calculate cost for this iteration
        cost[iter] = np.sum(abs(V - V_hat)**2)
        for i in range(num_sources):
            cost[iter] += config["H_sparsity"][i] * np.sum(H[i])

        # Stop iterations if change in cost function less than the tolerance
        if iter > 0 and cost[iter] < cost[iter-1] and cost[iter-1] - cost[iter] < config["tolerance"]:
            cost = cost[: iter] # trim vector
            break

    # Prepare the output
    if not is_W_cell:
        W = W[0]

    if not is_H_cell:
        H = H[0]

    if not is_P_cell:
        P = P[0]

    return [W, H, P, cost]