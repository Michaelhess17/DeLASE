from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error
import polars as pl

def get_shape_mode(λs):
    return stats.mode([λ.shape for λ in λs])

def filter_λs(λs, shape):
    return [λ for λ in λs if λ.shape == shape]

def analyze_lambda_convergence(λs, one_over_n_splits, eig=0, max_splits=None, max_poly_degree=3):
    n_splits, n_subjects = λs.shape
    
    if max_splits is None:
        max_splits = n_splits
    
    results = {}
    
    for subject in range(n_subjects):
        subject_results = {}
        
        # Prepare data
        X = one_over_n_splits[-max_splits:].reshape(-1, 1)  # Use the last max_splits points
        y = np.array([λs[i, subject][:, eig].mean() for i in range(-max_splits, 0)])  # Use the last max_splits points
        weights = 1/np.array([λs[i, subject][:, eig].std() for i in range(-max_splits, 0)])  # Use the last max_splits points
        
        # Perform weighted linear regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(X.flatten(), y, 1, w=weights)
            slope, intercept = coeffs
        
        # Calculate R-squared
        y_pred = slope * X.flatten() + intercept
        ss_tot = np.sum(weights * (y - np.mean(y))**2)
        ss_res = np.sum(weights * (y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard error of intercept
        n = len(X)
        x_bar = np.mean(X)
        s_xx = np.sum(weights * (X.flatten() - x_bar)**2)
        s_e = np.sqrt(np.sum(weights * (y - y_pred)**2) / (n - 2))
        intercept_std_err = s_e * np.sqrt(1/n + x_bar**2/s_xx)
        
        subject_results['linear'] = {
            'intercept': intercept,
            'intercept_std_err': intercept_std_err,
            'slope': slope,
            'r_squared': r_squared,
        }
        
        poly_results = get_polynomial_regression(X, y, weights, max_poly_degree)
        
        # Find the best polynomial model based on BIC
        best_poly = min(poly_results, key=lambda x: x['bic'])
        subject_results['best_polynomial'] = best_poly
        
        results[f'subject_{subject}'] = subject_results
    
    return results

def get_polynomial_regression(X, y, weights, max_poly_degree=3):
    poly_results = []
    for degree in range(1, max_poly_degree + 1):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y, sample_weight=weights)
        
        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred, sample_weight=weights)
        n_params = degree + 1
        n_samples = len(y)
        
        # Handle zero MSE case
        if mse == 0:
            bic = np.inf
        else:
            bic = n_samples * np.log(mse) + n_params * np.log(n_samples)
        
        poly_results.append({
            'degree': degree,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'bic': bic
        })
    return poly_results

def plot_lambda_convergence(λs, one_over_n_splits, results, max_splits=None, eig=0, subject=0):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    if max_splits is None:
        max_splits = λs.shape[0]
    
    y = np.array([λs[i, subject][:, eig].mean() for i in range(-max_splits, 0)])
    yerr = np.array([λs[i, subject][:, eig].std() for i in range(-max_splits, 0)])
    plt.scatter(x=one_over_n_splits[-max_splits:], y=y, s=50, c="k")
    plt.errorbar(one_over_n_splits[-max_splits:], y, yerr=yerr, fmt='none', capsize=5, alpha=0.75, c="k")
    
    # Plot linear regression
    linear_result = results[f'subject_{subject}']['linear']
    x_range = np.linspace(0, one_over_n_splits[-max_splits:].max(), 100)
    y_pred = linear_result['slope'] * x_range + linear_result['intercept']
    plt.plot(x_range, y_pred, color='r', label='Linear fit', lw=2)
    plt.errorbar(x=0.0, y=linear_result['intercept'], yerr=linear_result['intercept_std_err'], color='r', label='Intercept', lw=2)
    plt.scatter(x=0.0, y=linear_result['intercept'], c="r", s=50)

    plt.xlabel('1/N')
    plt.ylabel('λ')
    plt.title(f'Lambda Convergence for Subject {subject}')
    
    # Add text with regression results
    text = f"Linear fit:\ny-intercept = {linear_result['intercept']:.4f} ± {linear_result['intercept_std_err']:.4f}\n"
    text += f"R² = {linear_result['r_squared']:.4f}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(loc="upper right")
    plt.show()

def results_to_dataframe(results):
    data = []
    for subject, result in results.items():
        linear = result['linear']
        poly = result['best_polynomial']
        data.append({
            'Subject': subject,
            'Linear_Intercept': linear['intercept'],
            'Linear_Intercept_StdErr': linear['intercept_std_err'],
            'Linear_Slope': linear['slope'],
            'Linear_R_Squared': linear['r_squared'],
            'Best_Poly_Degree': poly['degree'],
            'Best_Poly_Intercept': poly['intercept'],
            'Best_Poly_BIC': poly['bic']
        })
    return pl.DataFrame(data)

def plot_eigvals_on_unit_circle(eigvals, c=None):
    if c is None:
        c = 'b'
    fig, ax = plt.subplots(figsize=(6, 6))
    ax1 = ax.scatter(eigvals.real, eigvals.imag, c=c, s=5)
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), c='k', linewidth=1)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Eigenvalues on the Unit Circle')
    fig.colorbar(ax1, ax=ax)
    plt.show()