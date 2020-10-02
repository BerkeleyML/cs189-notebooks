import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from ipywidgets import interactive
import ipywidgets as widgets
from ipywidgets import fixed
from sklearn.linear_model import LinearRegression




def generate_bool_widget(name):
    return widgets.RadioButtons(
    value=False,
    options=[True, False],
    description=name + ':',
    disabled=False
)

def generate_float_widget(name):
    return widgets.FloatSlider(
        value=0.7,
        min=0.,
        max=1.,
        step=0.01, 
        description=name + ': ', 
        continuous_update= False)

def generate_gamma_widget():
    return widgets.FloatSlider(
        value=0.7,
        min=0.,
        max=1.,
        step=0.01, 
        description='$\gamma$: ', 
        continuous_update= False)

def generate_int_widget(name, val, min_, max_, step=1):
    return widgets.IntSlider(
        value=val,
        min=min_,
        max=max_,
        step=step,
        description=name + ':',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')

def generate_d_widget():
    return widgets.IntSlider(
        value=401,
        min=201,
        max=3001,
        step=2,
        description='d: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')

def generate_s_widget():
    return widgets.IntSlider(
        value=100,
        min=1,
        max=200,
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
    

def plot_prediction(x_train, y_train, x_test, y_test, y_test_pred, show=True):
    ylim = [-2,2]
    plt.plot(x_test, y_test, label = 'True function')
    plt.scatter(x_train, y_train, marker='o', s=20, label = 'Training samples')
    plt.plot(x_test, y_test_pred, '-', ms=2, label = 'Learned function')
    plt.legend()
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

def solve_ls(phi, y, weights=None):
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

def solve(s, n, d, num_training_noise_seeds, 
          phi_type, x_type, f_type, 
          awgn_std, n_test, noise_seed_idx, 
          gamma, 
          plot_all=True):
    assert(d >= n)
    assert( d >= s)
    x_train = generate_x(x_type=x_type, n=n)
    phi_train = featurize(x_train, d, phi_type)
    y_train = generate_y(x=x_train, f_type=f_type)
    
    x_test= generate_x(x_type = 'uniform_random', n=n_test)
    y_test = generate_y(x=x_test, f_type=f_type)   
    phi_test = featurize(x_test, d, phi_type)
    
    weights = get_bilevel_weights(s,gamma,d)
    plot_weights(weights, gamma, s)
        
    lambd = s * (1 - gamma) / (n * gamma)
    SU = 1. / (1 + lambd)
    CNs_sqr = (n / d) * (lambd ** 2 / (1 + lambd) ** 2)
    CNe_sqr = awgn_std ** 2 * ((s / n) * ((1 + n * lambd ** 2 / d) / (1 + lambd) ** 2) + (n - s) / d)
    print("(1-SU)^2: {:.4f}, CNs^2: {:.4f}, CNe^2: {:.4f}".format(
            (1-SU)**2, CNs_sqr, CNe_sqr))

    noise = np.random.normal(0, awgn_std, size = [y_train.shape[0], num_training_noise_seeds])    
    y_train_noisy = y_train[:,None] + noise
    
    coeffs_noiseless, loss_noiseless  = solve_ls(phi_train, y_train, weights)
    true_coeffs = np.zeros_like(coeffs_noiseless)
    true_coeffs[:n] = solve_ls(phi_train[:,:n], y_train)[0]
    y_test_pred_noiseless = phi_test @ coeffs_noiseless
    if plot_all:
        plt.figure(figsize=[16, 9])
        plt.subplot(2,3,1)
        plot_prediction(x_train, y_train, x_test, y_test, y_test_pred_noiseless, show=not plot_all)   
        plt.subplot(2,3,4)
        plot_coeffs(coeffs_noiseless, true_coeffs, title = 'Signal only', show=not plot_all)
    pred_error_noiseless = np.mean((y_test_pred_noiseless - y_test)**2)
    print("Noiseless Training Loss: ", loss_noiseless)
    print("Noiseless Prediction Error: ", round(pred_error_noiseless,3))
          
    coeffs_noise, loss_noise  = solve_ls(phi_train, noise, weights)
    coeffs_noise = coeffs_noise.T
    true_coeffs_noise = np.zeros(d)
    y_test_pred_noise = phi_test @ coeffs_noise
    if plot_all:
        plt.subplot(2,3,2)
        plot_prediction(x_train, noise[:,noise_seed_idx], x_test, np.zeros_like(x_test), y_test_pred_noise[:,noise_seed_idx], show=not plot_all)
        plt.subplot(2,3,5)
        plot_coeffs(coeffs_noise[:,noise_seed_idx], true_coeffs_noise, title = 'Noise only', show=not plot_all)
    pred_error_noise = np.mean((y_test_pred_noise)**2)
    print("Pure Noise Training Loss: ", loss_noise)
    print("Pure Noise Prediction Error: ", round(pred_error_noise,3))
        
    coeffs = coeffs_noise + coeffs_noiseless[:,None]
    loss =  np.mean((phi_train @ coeffs - y_train_noisy)**2)
    y_test_pred = phi_test @ coeffs
 
    if plot_all:
        plt.subplot(2,3,3)
    plot_prediction(x_train, y_train_noisy[:,noise_seed_idx], x_test, y_test, y_test_pred[:,noise_seed_idx], show=not plot_all)
    
    if plot_all:
        plt.subplot(2,3,6)
    plot_coeffs(coeffs[:,noise_seed_idx], true_coeffs, title = 'Signal + Noise', show=not plot_all)
    pred_error = np.mean((y_test_pred - y_test[:,None])**2)
    print("Final Training Loss: ", loss)
    print("Final Prediction Error: ", round(pred_error,3))
    