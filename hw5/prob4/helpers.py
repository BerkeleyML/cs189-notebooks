import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from ipywidgets import interactive
import ipywidgets as widgets
from ipywidgets import fixed


def generate_gamma_widget(gamma=0.7, gamma_min=0.):
    return widgets.FloatSlider(
        value=gamma,
        min=gamma_min,
        max=1.,
        step=0.01, 
        description='$\gamma$: ', 
        continuous_update= False)

def generate_d_widget(d=401):
    return widgets.IntSlider(
        value=d,
        min=201,
        max=3001,
        step=2,
        description='d: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')

def generate_s_widget(s=100, max_s=200):
    return widgets.IntSlider(
        value=s,
        min=1,
        max=max_s,
        step=1,
        description='s: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')

def generate_awgn_std_widget():
    return widgets.FloatLogSlider(
        value=-1,
        base=np.sqrt(2),
        min=-20,
        max=4,
        description='$\sigma_{awgn}$: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='f')
        

def generate_x(n, x_type, x_low=-1, x_high=1):  
    if x_type == 'grid':
        x = np.linspace(x_low, x_high, n, endpoint = False).astype(np.float64)
    elif x_type == 'uniform_random':
        x = np.sort(np.random.uniform(x_low, x_high, n).astype(np.float64))
        #Note that for making it easy for plotting we sort the randomly sampled x in ascending order
    else:
        raise ValueError    
    return x


def mysign(x):
    y = np.sign(x)
    y[x == 0] = 1    
    return y


def generate_y(x, f_type):
    if f_type == 'x':
        y = x  
    elif f_type == 'cos2':
        y = np.cos(2*np.pi * x) 
    elif f_type == 'sign':
        y = mysign(x)        
    else: 
        raise ValueError     
    return y


def featurize_fourier(x, d, normalize = False):
    assert (d-1) % 2 == 0, "d must be odd"
    max_r = int((d-1)/2)
    n = len(x)
    A = np.zeros((n, d))
    A[:,0] = 1
    for d_ in range(1,max_r+1):
        A[:,2*(d_-1)+1] =  np.sin(d_*x*np.pi)
        A[:,2*(d_-1)+2] =  np.cos(d_*x*np.pi)
    if normalize:
        A[:,0] *= (1/np.sqrt(2))
        A *= np.sqrt(2)
    return A


def featurize(x, d, phi_type, normalize = True):
    function_map = {
#                 'polynomial':featurize_vandermonde, 
                    'fourier':featurize_fourier}
    return function_map[phi_type](x,d,normalize)


def get_bilevel_weights(s, gamma, d):
    weights = np.zeros(d)
    weights[:s] = np.sqrt(gamma*d/s)
    weights[s:] = np.sqrt((1-gamma)*d/(d-s))
    return weights
    

def plot_prediction(x_train, y_train, x_test, y_test, y_test_pred, show=True, title=''):
    ylim = [-2,2]
    plt.plot(x_test, y_test, label = 'True function')
    plt.scatter(x_train, y_train, marker='o', s=20, label = 'Training samples')
    plt.plot(x_test, y_test_pred, '-', ms=2, label = 'Learned function')
    plt.legend()
    plt.title(title)
    plt.ylim(ylim)
    if show:
        plt.show()
    
    
def plot_coeffs(coeffs, true_coeffs, title, show=True):
    ylim=[-1,1]
    plt.plot(coeffs, 'o--', label = 'Learned')
    plt.plot(true_coeffs, 'o--',  label='True')
    plt.ylabel('coeffs')
    plt.ylim(ylim)
    plt.xlabel('feature(k)')
    if title is not None:
        plt.title(title)
    plt.legend()
    if show:
        plt.show()    
    
def plot_weights(weights, gamma, s, show=True):
    ylim = [-1, 5]
    plt.plot(weights, 'o-')
    plt.ylim(ylim)
    plt.title('Bi-level weights, s = ' + str(s) + ', $\gamma$ = ' + str(round(gamma,3)))
    if show:
        plt.show()
        
from sklearn.linear_model import Ridge

def solve_ridge(phi, y, lambda_ridge, weights = None):
    if weights is None:
        weights  = np.ones(phi.shape[1]) 
#     weights = np.ones(phi.shape[1])
    phi_weighted = weights*phi
    Rdg = Ridge(fit_intercept=False, normalize=False, alpha = lambda_ridge)

    ### start c2 ###
    Rdg.fit(phi_weighted, y)
    coeffs_weighted  = Rdg.coef_
    alpha = coeffs_weighted*weights

    ### end c2 ###
    
    loss = np.mean((y - phi @ alpha.T)**2) + lambda_ridge*np.mean(coeffs_weighted**2)
    return alpha, loss       

from sklearn.linear_model import LinearRegression

def solve_ls2(phi, y, weights=None):
    d = phi.shape[1]
    if weights is None:
        weights  = np.ones(d) 
    phi_weighted = weights*phi
    LR = LinearRegression(fit_intercept=False, normalize=False)
    LR.fit(phi_weighted, y)
    coeffs_weighted = LR.coef_
    alpha = coeffs_weighted*weights
    loss = np.mean((y - phi @ alpha.T)**2)
    return alpha, loss


def solve2(s, n, d, num_training_noise_seeds, 
          phi_type, x_type, f_type, 
          awgn_std, n_test, noise_seed_idx, 
          gamma, 
          plot_all=True):
    # TODO: print SU, CNs, CNe values based on params
    assert(d >= n)
    assert( d >= s)
    x_train = generate_x(x_type=x_type, n=n)
    phi_train = featurize(x_train, d, phi_type)
    y_train = generate_y(x=x_train, f_type=f_type)
    
    x_test= generate_x(x_type = 'uniform_random', n=n_test)
    y_test = generate_y(x=x_test, f_type=f_type)   
    phi_test = featurize(x_test, d, phi_type)
    
    weights = get_bilevel_weights(s,gamma,d)
        
    # Expected prediction error
    lambd = s * (1 - gamma) / (n * gamma)
    SU = 1. / (1 + lambd)
    CNs_sqr = (n / d) * (lambd ** 2 / (1 + lambd) ** 2)
    CNe_sqr = awgn_std ** 2 * ((s / n) * ((1 + n * lambd ** 2 / d) / (1 + lambd) ** 2) + (n - s) / d)
    print("(1-SU)^2: {:.4f}, CNs^2: {:.4f}, CNe^2: {:.4f}".format(
            (1-SU)**2, CNs_sqr, CNe_sqr))
    print("lambda: "+str(round(lambd * n,4)))

    # Generate noise
    noise = np.random.normal(0, awgn_std, size = [y_train.shape[0], num_training_noise_seeds])    
    y_train_noisy = y_train[:,None] + noise
    
    phi_ridge = phi_train[:,:s]
    lambda_ridge = lambd * n

    coeffs_ridge, loss_ridge = solve_ridge(phi_ridge, y_train_noisy, lambda_ridge, weights = np.ones(s)) 
    y_test_pred_ridge = phi_test[:,:s] @ coeffs_ridge.T
    
    coeffs, loss  = solve_ls2(phi_train, y_train_noisy, weights)
    y_test_pred = phi_test @ coeffs.T
    
    true_coeffs = np.zeros(d)
    true_coeffs[:n] = solve_ls2(phi_train[:,:n], y_train)[0]
            
    if plot_all:
        plt.figure(figsize=[16, 9])

        plt.subplot(2,2,1)
    plot_prediction(x_train, y_train_noisy[:,noise_seed_idx], x_test, y_test, y_test_pred[:,noise_seed_idx], show=not plot_all, title="Overparameterized Prediction")
    
    if plot_all:
        plt.subplot(2,2,2)
    plot_coeffs(coeffs[noise_seed_idx,:], true_coeffs, title = 'Overparameterized Coefficients', show=not plot_all)
    pred_error = np.mean((y_test_pred - y_test[:,None])**2)
    
    if plot_all:
        plt.subplot(2,2,3)
    plot_prediction(x_train, y_train_noisy[:,noise_seed_idx], x_test, y_test, y_test_pred_ridge[:,noise_seed_idx], show=not plot_all, title="Ridge Prediction")
   
    if plot_all:
        plt.subplot(2,2,4)
    plot_coeffs(coeffs_ridge[noise_seed_idx,:], true_coeffs, title = 'Ridge Coefficients', show=not plot_all)
    plt.show()
    
    
    print("Final Training Loss: ", loss)   
    print("Final Prediction Error: ", round(pred_error,3))
    
    plot_weights(weights, gamma, s)

    
def solve3(s, n, d, num_training_noise_seeds, 
          phi_type, x_type, f_type, 
          awgn_std, n_test, noise_seed_idx, 
          gamma, 
          plot_all=True):
    # TODO: print SU, CNs, CNe values based on params
    assert(d >= n)
    assert( d >= s)
    x_train = generate_x(x_type=x_type, n=n)
    phi_train = featurize(x_train, d, phi_type)
    y_train_true = generate_y(x=x_train, f_type=f_type)
    y_train = np.zeros(n)
    y_train[n//2] = 1
    x_test= generate_x(x_type = 'uniform_random', n=n_test)
    y_test = generate_y(x=x_test, f_type=f_type)   
    phi_test = featurize(x_test, d, phi_type)
    
    weights = get_bilevel_weights(s,gamma,d)
#     plot_weights(weights, gamma, s)
        
    # Expected prediction error
#     lambd = s * (1 - gamma) / (n * gamma)
#     SU = 1. / (1 + lambd)
#     CNs_sqr = (n / d) * (lambd ** 2 / (1 + lambd) ** 2)
#     CNe_sqr = awgn_std ** 2 * ((s / n) * ((1 + n * lambd ** 2 / d) / (1 + lambd) ** 2) + (n - s) / d)
#     print("(1-SU)^2: {:.4f}, CNs^2: {:.4f}, CNe^2: {:.4f}".format(
#             (1-SU)**2, CNs_sqr, CNe_sqr))
#     print("lambda: "+str(round(lambd,4)))

    # Solve the noiseless case
    coeffs_impulse, _  = solve_ls2(phi_train, y_train, weights)
    y_test_pred_impulse = phi_test @ coeffs_impulse
    
    
    phi_train_lf = phi_train[:, :s]
    
    coeffs_impulse_lf, _  = solve_ls2(phi_train_lf, y_train, weights[:s])
    y_test_pred_impulse_lf = phi_test[:,:s] @ coeffs_impulse_lf
        
        
    coeffs, loss  = solve_ls2(phi_train, y_train_true, weights)
    y_test_pred = phi_test @ coeffs.T
    
    true_coeffs = np.zeros(d)
    true_coeffs[:n] = solve_ls2(phi_train[:,:n], y_train_true)[0]
            
    if plot_all:
        plt.figure(figsize=[16, 6])

        plt.subplot(1,2,2)
    plot_prediction(x_train, y_train_true, x_test, y_test, y_test_pred, show=not plot_all)
    
#     phi_train_hf = phi_train[:, s:]
#     coeffs_impulse_hf, _  = solve_ls(phi_train_hf, y_train, weights[s:])
#     y_test_pred_impulse_hf = phi_test[:,s:] @ coeffs_impulse_hf
   
#     print('all:',np.mean(y_test_pred_impulse ** 2))
#     print('LF:',np.mean(y_test_pred_impulse_lf ** 2))
#     print('HF:',np.mean(y_test_pred_impulse_hf ** 2))
    if plot_all:
#         plt.figure(figsize=[10, 6])

        plt.subplot(1,2,1)
        plt.title('Impulse response')

        plt.plot(x_test, np.abs(y_test_pred_impulse), '-', label = 'All features')
#         plt.show()
        plt.plot(x_test, np.abs(y_test_pred_impulse_lf), '-', label = 'First s low frequency features')
        plt.yscale('log')
        plt.ylim([1e-3, 1.1])
#         plt.plot([-1,1], [2e-2,2e-2], '--')
        plt.legend()
        plt.show()
#         plt.figure(figsize=[16, 9])
#         plt.plot(x_test, np.abs(y_test_pred_impulse_hf), '-')
#         plt.ylim([1e-3, 1.1])
#         plt.yscale('log')
#         plt.plot([-1,1], [2e-2,2e-2], '--')
#         plt.show()
      

