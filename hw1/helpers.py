import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def make_random_system(n_states, n_inputs, z_min=0.0, z_max=1.0, must_have_real_eigenvalues=[], 
                       must_have_complex_eigenvalues=[], ccf=False):
    """
    make_random_system: make a random A and B matrix, whose eigenvalues $z$ lie in the annulus 
    $z_{min}<=|z|<=z_{max}$, occurring as complex conjugate pairs, plus at most one purely real. 
    Additionally, a number of "must-have" eigenvalues can be specified that the system is guaranteed to have, 
    which need not lie in the annulus.
    
    Specify ccf=True if you want the system to be given in controllable canonical form.
    
    arguments:
        n_states: the number of states you want the sytem to have.
        n_inputs: the number of inputs you want the system to have.
        z_min: the smallest allowable magnitude for the random eigenvalues.
        z_max: the largest allowable magnitude for the random eigenvalues.
        must_have_real_eigenvalues: a list of purely eigenvalues that the random system is guaranteed to have.
        must_have_complex_eigenvalues: a list of complex conjugate eigenvalues that the random system is guaranteed to have.
            You only need to specify one member of the conjugate pair-- the other is implied.
        ccf: whether or not you want the system matrices to be returned in controllable canonical form.
    
    returns:
        a_matrix: A-matrix of the random system.
        b_matrix: B-matrix of the random system.
    """
    n_random = n_states - np.size(must_have_real_eigenvalues) - 2*np.size(must_have_complex_eigenvalues)
    assert n_random > 0, 'no random eigenvalues'
    n_real = n_random % 2
    n_conj_pairs = int(np.floor(n_random/2))
    # This weird probability distribution over the eigenvalue magnitudes ensures that the eigenvalues will be distributed uniformly over the annulus.
    z_mags = np.sqrt(z_min**2+np.random.rand(n_conj_pairs)*(z_max**2-z_min**2))
    z_angles = 2*np.pi*np.random.rand(n_conj_pairs)
    z_random_complex = z_mags * np.exp(1j*z_angles)
    z_complex = np.concatenate((z_random_complex, must_have_complex_eigenvalues))
    z_real = must_have_real_eigenvalues
    if n_real != 0:
        z_random_real = z_min + np.random.rand(n_real)*(z_max-z_min)
        z_real = np.concatenate((z_real, z_random_real))
    if ccf:
        z = np.concatenate((z_complex, z_real))
        a_matrix, b_matrix = make_ccf_system_from_eigenvalues(z)
    else:
        a_matrix, b_matrix = make_random_system_from_eigenvalues(z_real, z_complex)
    return a_matrix, b_matrix

def plot_eigenvalues(a_matrices, labels=None):
    """
    plot_poles: given an list A matrix, find their eigenvalues and plot them on the complex plane. 
    Additionally, the unit circle is plotted as a dashed line. Additionally, each A matrix can be given a label, 
    which will be used to identify that matrix's eigenvalues in a legend.
    
    arguments:
        a_matrices: the system A-matrices.
        labels: list of legend entries, one for each A matrix.
    
    returns:
        None
    """
    markers=['o','x','1','8','*','+'];
    fig = plt.figure(figsize=(8, 8))    
    for i, a_matrix in enumerate(a_matrices):
        eigenvalues = np.linalg.eigvals(a_matrix)
        if labels:
            label = labels[i]
        else:
            label=''
        plt.plot(np.real(eigenvalues), np.imag(eigenvalues), linestyle='None', marker=markers[i], ms=5, label=label)
    if labels:
        plt.legend(loc='upper right')
    t = np.linspace(0, 2*np.pi, 1000)
    plt.plot(np.cos(t),np.sin(t), 'k--')
    plt.grid(True)
    return None


def identify_system(state_trace, inpt):
    """
    identify_system: given an input and the state trace it generated, determine the A and B matrices 
                     of the system that was used to generate the state trace.
    
    arguments:
        state_trace: the state data.
        inpt: the input data:
        
    returns:
        a_identified: the A matrix identified from the state and input data (numpy matrix)
        b_identified: the B matrix identified from the state and input data (numpy matrix)
    """

    # set up least-squares equation from state and input data
    n_states = np.shape(state_trace)[1]
    n_inputs = np.shape(inpt)[1]
    lsq_b = np.matrix(state_trace[1:,:])
    lsq_a = np.concatenate((state_trace[:-1,:],inpt),axis=1)
    # solve the least-squares problem. Output should be [A B]^T
    # YOUR CODE HERE
    # Based on the code around it, it looks like your code needs to produce two variables called a_identified
    # and b_identified.
    ab_identified, residuals, rank, s = np.linalg.lstsq(lsq_a, lsq_b, rcond=None)
    # What else does your code need to do? Well, let's look at the documentation. You need to return two variables,
    # a_identified and b_identified, and they both need to be numpy matrices.
    a_identified = ab_identified[:n_states,:]
    a_identified = np.matrix(a_identified).T
    b_identified = ab_identified[n_states:,:]
    b_identified = np.matrix(b_identified).T
    
    return a_identified, b_identified


def random_input(t, n_inputs, mean=0, std=1):
    """
    random_input: given a time sequence t, generate a random input trace. Each entry of the trace will be sampled
                  from a Gaussian distribution with the given mean and standard deviation.
    """
    k = np.shape(t)[0]
    random_trace = np.random.normal(loc=mean, scale=std, size=(k, n_inputs))
    random_trace = np.matrix(random_trace)
    return random_trace


def make_state_trace(a_matrix, b_matrix, inpt, initial_state=None):
    """
    make_state_trace: for a given A and B matrices, initial condition, and input, "run" the system and 
    calculate the system state for each time step. The length of the trace will be the length of the input plus one.
    
    arguments:
        a_matrix: A matrix of the system (numpy matrix, n by n)
        b_matrix: B matrix of the system (numpy matrix, p by n)
        inpt: input (u) (numpy matrix, T by p)
              initial_state: initial state for the system (numpy matrix, n by 1). If no initial state is provided, 
              the initial state will be the origin.
    
    returns:
        state_trace: a numpy matrix/ndarray containing the state at 
                     each time step that was run (numpy ndarray, T+1 by n).
    """
    n_states = np.shape(a_matrix)[0]
    n_timesteps = np.shape(inpt)[0]
    state_trace = np.zeros((n_timesteps+1, n_states))
    if initial_state:
        x0 = initial_state
    else:
        x0 = np.matrix(np.zeros((n_states, 1)))
    state_trace[0,:] = np.array(x0.T)
    for i in range(n_timesteps):
        current_state = np.matrix(state_trace[i,:]).T
        current_inpt = np.matrix(inpt[i]).T
        next_state = a_matrix*current_state + b_matrix*current_inpt
        state_trace[i+1,:] = np.array(next_state.T)
    return state_trace

def random_input(t, n_inputs, mean=0, std=1):
    """
    random_input: given a time sequence t, generate a random input trace. Each entry of the trace will be sampled
                  from a Gaussian distribution with the given mean and standard deviation.
    """
    k = np.shape(t)[0]
    random_trace = np.random.normal(loc=mean, scale=std, size=(k, n_inputs))
    random_trace = np.matrix(random_trace)
    return random_trace

def plot_eigenvalues(a_matrices, labels=None):
    """
    plot_poles: given an list A matrix, find their eigenvalues and plot them on the complex plane. 
    Additionally, the unit circle is plotted as a dashed line. Additionally, each A matrix can be given a label, 
    which will be used to identify that matrix's eigenvalues in a legend.
    
    arguments:
        a_matrices: the system A-matrices.
        labels: list of legend entries, one for each A matrix.
    
    returns:
        None
    """
    markers=['o','x','1','8','*','+'];
    fig = plt.figure(figsize=(8, 8))    
    for i, a_matrix in enumerate(a_matrices):
        eigenvalues = np.linalg.eigvals(a_matrix)
        if labels:
            label = labels[i]
        else:
            label=''
        plt.plot(np.real(eigenvalues), np.imag(eigenvalues), linestyle='None', marker=markers[i], ms=5, label=label)
    if labels:
        plt.legend(loc='upper right')
    t = np.linspace(0, 2*np.pi, 1000)
    plt.plot(np.cos(t),np.sin(t), 'k--')
    plt.grid(True)
    return None