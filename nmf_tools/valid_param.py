import numpy as np
from .utils import *



def  ValidateParameters(config_in, V, num_basis_elems, ):

    config_out = dict(config_in)
    m, n = V.shape()

    is_H_cell = False
    is_W_cell = False

    num_sources = len(num_basis_elems)

    # Initialize encoding matrices
    if "H_init" not in config_out.keys() or not config_out["H_init"]: # not given any inital encoding matrices. Fill these in.
        if num_sources == 1:
            is_H_cell = False
        else:
            is_H_cell = True

        config_out["H_init"] = [None for _ in range(num_sources)]

        for i in range(num_sources):
            config_out["H_init"][i] = np.max(np.random.random([num_basis_elems[i], n]), eps)
    elif isinstance(config_out["H_init"], list) and len(config_out["H_init"]) != num_sources:  # given an incorrect number of initial encoding matrices
        raise Exception("Requested %d sources. Given %d initial encoding matrices." % (num_sources, len(config_out["H_init"])))
    elif not isinstance(config_out["H_init"], list): # given a matrix
        is_H_cell = False
        config_out["H_init"] = [config_out["H_init"]]
    else: # organize encoding matrices as {H_1 H_2 ... H_num_bases}
        is_H_cell = True
        config_out["H_init"] = config_out["H_init"][:]

    # Initialize basis matrices
    if not isfield(config_out, 'W_init') or isempty(config_out["W_init"]): # not given any inital basis matrices. Fill these in.
        if num_sources == 1:
            is_W_cell = false
        else:
            is_W_cell = true
        config_out["W_init"] = cell(1, num_sources)

        for i in range(num_sources):
            config_out["W_init"][i] = max(rand(m, num_basis_elems[i]), eps)
            config_out["W_init"][i] = config_out["W_init"][i] * diag(1 / sqrt(sum(config_out["W_init"][i] ** 2, 1)))

    elif iscell(config_out["W_init"]) and length(config_out["W_init"]) != num_sources: # given an incorrect number of initial basis matrices
        error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out["W_init"])), ' initial basis matrices.'])
    elif not iscell(config_out["W_init"]): # given a matrix
        is_W_cell = false
        config_out["W_init"] = [config_out["W_init"]]
    else: # organize basis matrices as {W_1 W_2 ... W_num_bases}
        is_W_cell = true
        config_out["W_init"] = config_out["W_init"][:].H

    # Sparsity levels for basis matrices
    if not isfield(config_out, 'W_sparsity') or isempty(config_out["W_sparsity"]): # not given a sparsity level. Fill this in.
        config_out["W_sparsity"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["W_sparsity"][i] = 0
    elif iscell(config_out["W_sparsity"]) and length(config_out["W_sparsity"]) > 1 and length(config_out["W_sparsity"]) != num_sources: # given an incorrect number of sparsity levels
        error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out["W_sparsity"])), ' sparsity levels.'])
    elif not iscell(config_out["W_sparsity"]) or length(config_out["W_sparsity"]) == 1: # ext one sparsity level to all basis matrices
        if iscell(config_out["W_sparsity"]):
            temp = max(config_out["W_sparsity"][1], 0)
        else:
            temp = max(config_out["W_sparsity"], 0)
        config_out["W_sparsity"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["W_sparsity"][i] = temp
    else: # make sure all given sparsity levels are non-negative
        for i in range(num_sources):
            config_out["W_sparsity"][i] = max(config_out["W_sparsity"][i], 0)

    # Sparsity levels for encoding matrices
    if not isfield(config_out, 'H_sparsity') or isempty(config_out["H_sparsity"]): # not given a sparsity level. Fill this in.
        config_out["H_sparsity"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["H_sparsity"][i] = 0

    elif iscell(config_out["H_sparsity"]) and length(config_out["H_sparsity"]) > 1 and length(config_out["H_sparsity"]) != num_sources: # given an incorrect number of sparsity levels
        error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out["H_sparsity"])), ' sparsity levels.'])
    elif not iscell(config_out["H_sparsity"])  or length(config_out["H_sparsity"]) == 1: # ext one sparsity level to all encoding matrices
        if iscell(config_out["H_sparsity"]):
            temp = max(config_out["H_sparsity"][0], 0)
        else:
            temp = max(config_out["H_sparsity"], 0)
        
        config_out["H_sparsity"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["H_sparsity"][i] = temp
    else:  # make sure all given sparsity levels are non-negative
        for i in range(num_sources):
            config_out["H_sparsity[i]"] = max(config_out["H_sparsity"][i], 0)
        
    

    # Update switches for basis matrices
    if not isfield(config_out, 'W_fixed') or isempty(config_out["W_fixed"]):  # not given an update switch. Fill this in.
        config_out["W_fixed"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["W_fixed"][i] = false
        
    elif iscell(config_out["W_fixed"]) and  length(config_out["W_fixed"]) > 1 and  length(config_out["W_fixed"]) != num_sources:  # given an incorrect number of update switches
        error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out["W_fixed"])), ' update switches.'])
    elif not iscell(config_out["W_fixed"])  or length(config_out["W_fixed"]) == 1:  # ext one update switch level to all basis matrices
        if iscell(config_out["W_fixed"]):
            temp = config_out["W_fixed"][0]
        else:
            temp = config_out["W_fixed"]
        
        config_out["W_fixed"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["W_fixed"][i] = temp

    # Update switches for encoding matrices
    if not isfield(config_out, 'H_fixed') or isempty(config_out["H_fixed"]):  # not given an update switch. Fill this in.
        config_out["H_fixed"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["H_fixed"][i] = false
        
    elif iscell(config_out["H_fixed"]) and  length(config_out["H_fixed"]) > 1 and  length(config_out["H_fixed"]) != num_sources:  # given an incorrect number of update switches
        error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out["H_fixed"])), ' update switches.'])
    elif not iscell(config_out["H_fixed"])  or length(config_out["H_fixed"]) == 1:  # ext one update switch level to all encoding matrices
        if iscell(config_out["H_fixed"]):
            temp = config_out["H_fixed"][0]
        else:
            temp = config_out["H_fixed"]
        
        config_out["H_fixed"] = cell(num_sources, 1)
        for i in range(num_sources):
            config_out["H_fixed[i]"] = temp

    # Maximum number of update iterations
    if not isfield(config_out, 'maxiter') or config_out["maxiter"] <= 0:
        config_out["maxiter"] = 100
    

    # Maximum tolerance in cost function change per iteration
    if not isfield(config_out, 'tolerance') or config_out["tolerance"] <= 0:
        config_out["tolerance"] = 1e-3
    
    return [config_out, is_W_cell, is_H_cell]