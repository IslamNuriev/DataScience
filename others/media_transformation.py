import itertools
import logging
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.api as sm
from tqdm import tqdm


def upload_media_params_file(file_path: str) -> pd.DataFrame:
    """Uploads media parameters from an Excel file. 

    Args:
        file_path: Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing media parameters.
    """
    media_params = pd.read_excel(file_path, sheet_name='data')
    return media_params


def prepare_media_params_init(df: pd.DataFrame) -> dict:
    """
    Return dictionary of media parameters initialization values.

    Args:
        df (pd.DataFrame): DataFrame containing media parameters.

    Returns:
        dict: Dictionary containing initialization values for media parameters.
    """
    params_names = [
                    'decay_param',
                    'alpha_param',
                    'beta_param',
                    'gamma_param',
                ]

    initializiation_values = dict()

    initializiation_values['media'] = list(df['media'])

    params = np.array([])
    for param in params_names:
        params = np.concatenate(
            (
                params,
                df[param].to_numpy()
            )
        )
    initializiation_values['params'] = list(params)

    decay_t = tuple(
        zip(
            df['decay_bound_min'].to_numpy(),
            df['decay_bound_max'].to_numpy(),
        )
    )
    
    alpha_t = tuple(
        zip(
            df['alpha_bound_min'].to_numpy(),
            df['alpha_bound_max'].to_numpy(),
        )
    )

    beta_t = tuple(
        zip(
            df['beta_bound_min'].to_numpy(),
            df['beta_bound_max'].to_numpy(),
        )
    )

    gamma_t = tuple(
        zip(
            df['gamma_bound_min'].to_numpy(),
            df['gamma_bound_max'].to_numpy(),
        )
    )
    bounds_list = [decay_t, alpha_t, beta_t, gamma_t]

    bounds_temp = list()
    for bounds in bounds_list:
        bounds_temp.extend(list(bounds))
    bounds_result = tuple(bounds_temp)
    
    initializiation_values['bounds'] = bounds_result

    return initializiation_values


def select_media_params(df: pd.DataFrame, media_params: List) -> pd.DataFrame:
    """
    Selects media parameters from a DataFrame based on given media_params.

    Args:
        df (pd.DataFrame): DataFrame containing media parameters.
        media_params (List): List of media parameters to select.

    Returns:
        pd.DataFrame: DataFrame containing selected media parameters.
    """
    return df.loc[df['media'].isin(media_params)]


def exclude_media_params(df: pd.DataFrame, media_params: List) -> pd.DataFrame:
    """
    Excludes media parameters from a DataFrame based on given media_params.

    Args:
        df (pd.DataFrame): DataFrame containing media parameters.
        media_params (List): List of media parameters to exclude.

    Returns:
        pd.DataFrame: DataFrame containing media parameters excluding the given media_params.
    """
    return df.loc[~df['media'].isin(media_params)]


def add_media_params(df: pd.DataFrame, new_media_params: Dict) -> pd.DataFrame:
    """
    Adds new media parameters to a DataFrame.

    Before using this function, you should create a dictionary with the following structure
    and pass it as an argument to the function.

    Args:
        df (pd.DataFrame): DataFrame containing existing media parameters.
        new_media_params (Dict): Dictionary containing new media parameters to add.

    Returns:
        pd.DataFrame: DataFrame containing the original media parameters plus the new ones.
    
    If you still want to add manually:
    new_media_params = {
        'media': 'NEW', 
        'decay_param': 0.25, 
        'decay_bound_min': 0.1, 
        'decay_bound_max': 0.8,
        'alpha_param': 3.0, 
        'alpha_bound_min': 0.1, 
        'alpha_bound_max': 3.0,
        'beta_param': 0.5,
        'beta_bound_min': 0.1,
        'beta_bound_max': 2.0,
        'gamma_param': 2.0, 
        'gamma_bound_min': 0.1,
        'gamma_bound_max': 3.0,
    }
    """
    new_media_params_df = pd.DataFrame(pd.Series(new_media_params)).transpose()
    
    return pd.concat([df, new_media_params_df], ignore_index=True)


def saturate_power(series: pd.Series, alpha: float, beta: float, gamma: float) -> pd.Series:
    '''
    Saturates a series using a power function.

    Parameters:
    - series (pd.Series): Input series to be saturated.
    - alpha (float): Scaling parameter. The shape is the same, but absolute numbers on the Y-axis are multiplied by alpha.
    - beta (float): Determines how fast the function reaches the plateau. Closer to 0.1 - rapid growth, closer to 1.0 - slower.
    - gamma (float): Shape of transformation function. If lower than 1 - closer to a C-shape. If more than 1.0, then closer to an S-shape.

    Returns:
    - pd.Series: Saturated series.
    '''
    saturated_media = np.where(
        np.array(series) == 0,
        0,
        pd.Series(alpha / (1 + beta * (np.power(series, -gamma))))
    )
    return pd.Series(saturated_media, index=series.index).fillna(0)


def find_saturated_point_power(media_spends: float, alpha: float, beta: float,
                               gamma: float) -> float:
    '''
    Calcualate point of saturation for specific point using a Power function.

    Parameters:
    - media_spends (float): Media-spends original before saturation transformation.
    - alpha (float): Scaling parameter. The shape is the same, but absolute numbers on the Y-axis are multiplied by alpha.
    - beta (float): Determines how fast the function reaches the plateau. Closer to 0.1 - rapid growth, closer to 1.0 - slower.
    - gamma (float): Shape of transformation function. If lower than 1 - closer to a C-shape. If more than 1.0, then closer to an S-shape.

    Returns:
    - float: Saturated value.
    '''
    if media_spends == 0:
        return 0
    saturation_point = alpha / (1 + beta * media_spends ** -gamma)
    return saturation_point


def saturate_hill(series: pd.Series, alpha: float, beta: float, gamma: float) -> pd.Series:
    """
    Saturates a series using a Hill function.

    Parameters:
    - series (pd.Series): Input series to be saturated.
    - alpha (float): Scaling parameter. The shape is the same, but absolute numbers on the Y-axis are multiplied by alpha.
    - beta (float): Determines how fast the function reaches the plateau. Closer to 0.1 - rapid growth, closer to 1.0 - slower.
    - gamma (float): Shape of transformation function. If lower than 1 - closer to a C-shape. If more than 1.0, then closer to an S-shape.

    Returns:
    - pd.Series: Saturated series.
    """
    saturated_media = np.where(
        np.array(series) == 0,
        0,
        pd.Series(alpha * np.power(series, beta) / (gamma + np.power(series, beta)))
    )
    return pd.Series(saturated_media, index=series.index).fillna(0)


def find_saturated_point_hill(media_spends: float, alpha: float, beta: float,
                              gamma: float) -> float:
    """
    Calcualate point of saturation for specific point using a Hill function.

    Parameters:
    - media_spends (float): Media-spends original before saturation transformation.
    - alpha (float): Scaling parameter. The shape is the same, but absolute numbers on the Y-axis are multiplied by alpha.
    - beta (float): Determines how fast the function reaches the plateau. Closer to 0.1 - rapid growth, closer to 1.0 - slower.
    - gamma (float): Shape of transformation function. If lower than 1 - closer to a C-shape. If more than 1.0, then closer to an S-shape.

    Returns:
    - float: Saturated value.
    """
    if media_spends == 0:
        return 0
    saturation_point = alpha * media_spends ** beta / (gamma + media_spends ** beta)
    return saturation_point


def adstock(series: pd.Series, decay_rate: float) -> pd.Series:
    """
    Apply adstock transformation to a series of advertising data.

    Parameters:
    - series (pd.Series): Series of advertising data (e.g., spend, impressions).
    - decay_rate (float): Decay rate for adstock effect (between 0 and 1).

    Returns:
    - pd.Series: Transformed series with adstock applied.
    """
    adstocked = [series.iloc[0]]

    for i in range(1, len(series)):
        adstocked_value = series.iloc[i] + decay_rate * adstocked[i-1]
        adstocked.append(adstocked_value)

    return pd.Series(adstocked, index=series.index).fillna(0)


def create_media_params_dict(params: pd.DataFrame, media_factors: List[str]):
    """
    Create a dictionary of media parameters.

    Parameters:
    - params (list): List of parameter values.
    - media_factors (list): List of media factors.

    Returns:
    - dict: Dictionary containing media parameters.
    """
    media_params = {}
    n = len(media_factors)
    for i, key in enumerate(media_factors):
        media_params[key] = {
            'decay': params[i::n][0],
            'alpha': params[i::n][1],
            'beta': params[i::n][2],
            'gamma': params[i::n][3]
        }
    return media_params


def transform_media_data(df: pd.DataFrame, params: Dict[str, Dict[str, float]], media_factors: List[str],
                         is_adstock: bool = True, is_saturate: bool = True,
                         saturate_func: str = 'saturate_hill',
                         is_transform_df: bool = False) -> pd.DataFrame:
    """
    Transform media data based on specified parameters and functions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing media data.
    - params (dict): Dictionary containing media parameters.
    - media_factors (list): List of media factors.
    - is_adstock (bool): Flag to indicate whether to apply adstock transformation (default is True).
    - is_saturate (bool): Flag to indicate whether to apply saturation transformation (default is True).
    - saturate_func (str): Name of the saturation function to use ('saturate_hill' or 'saturate_power', default is 'saturate_hill').
    - is_transform_df (bool): Consequent transformation if needed.

    Returns:
    - pd.DataFrame: Transformed media data DataFrame.
    """
    media_transformed = pd.DataFrame()

    if is_transform_df:
        order = params['media_combination'].split('->')
    else:
        order = media_factors.copy()

    for media_factor in order:
        if is_adstock:
            media_transformed[media_factor] = adstock(
                df[media_factor],
                params.get(media_factor).get('decay')
            )
        else:
            media_transformed[media_factor] = df[media_factor].copy()

    for media_factor in order:
        if is_saturate:
            if saturate_func == 'saturate_hill':
                saturate_function = saturate_hill
            elif saturate_func == 'saturate_power':
                saturate_function = saturate_power

            media_transformed[media_factor] = saturate_function(
                    media_transformed[media_factor],
                    params.get(media_factor).get('alpha'),
                    params.get(media_factor).get('beta'),
                    params.get(media_factor).get('gamma'),
            )

    return media_transformed


def lag_data(df: pd.DataFrame, columns_to_lag: List[str], shift: int = 1) -> pd.DataFrame:
    """
    Lag the data in specified columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - columns_to_lag (List[str]): List of column names to lag.
    - shift (int): Number of periods to shift the data (default is 1).

    Returns:
    - pd.DataFrame: DataFrame with lagged data.
    """
    df_copy = df.copy()
    for col in columns_to_lag:
        name_new = f'{col}_lag_{shift}'
        df_copy[name_new] = df_copy[col].shift(shift)
    df_copy.dropna(inplace=True)
    return df_copy


def optimize_media_params(y: pd.Series, df: pd.DataFrame, media_factors: List[str],
                          params_init: List[float], bounds_init: List[Tuple[float, float]],
                          epsilon_step: float = 1e-08, is_adstock: bool = True, is_saturate: bool = True,
                          saturate_function: str = 'saturate_power', tol: float = 1e-08,
                          non_media_factors: List[str] = [],
                          optimization_method: str = 'L-BFGS-B') -> List[float]:
    """
    Optimize media parameters.

    Parameters:
    - y (pd.Series): Series of target variable.
    - df (pd.DataFrame): DataFrame containing predictor variables.
    - media_factors (List[str]): List of media factors.
    - params_init (List[float]): Initial guess for the parameters to be optimized.
    - bounds_init (List[Tuple[float, float]]): List of tuples defining the bounds for each parameter.
    - epsilon_step (float): Step size for numerical differentiation (default is 1e-08).
    - is_adstock (bool): Flag indicating whether to apply adstock transformation (default is True).
    - is_saturate (bool): Flag indicating whether to apply saturation transformation (default is True).
    - saturate_function (str): Name of the saturation function to use ('saturate_hill' or 'saturate_power', default is 'saturate_power').
    - tol (float): Tolerance for optimization convergence (default is 1e-08).
    - non_media_factors (List[str]): List of non-media factors to include in the model (default is []).
    - optimization_method (str): Method for SciPy optimization. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html 

    Returns:
    - List[float]: Optimized parameters.
    """

    def rss(params_init: List[float], y: pd.Series, df: pd.DataFrame,
            media_factors: List[str], is_adstock: bool, is_saturate: bool,
            saturate_function: str) -> float:
        """
        Compute the residual sum of squares of function given on specific Data Frame.

        Parameters:
        - params_init (List[float]): Initial guess for the parameters to be optimized.
        - y (pd.Series): Series of target variable.
        - df (pd.DataFrame): DataFrame containing predictor variables.
        - media_factors (List[str]): List of media factors.
        - is_adstock (bool): Flag indicating whether to apply adstock transformation.
        - is_saturate (bool): Flag indicating whether to apply saturation transformation.
        - saturate_function (str): Name of the saturation function to use ('saturate_hill' or 'saturate_power').

        Returns:
        - float: Residual sum of squares.
        """

        params_dict = create_media_params_dict(params_init, media_factors)
        df_transformed = transform_media_data(
            df,
            params_dict,
            media_factors,
            is_adstock=is_adstock,
            is_saturate=is_saturate,
            saturate_func=saturate_function
        )

        if len(non_media_factors) > 0:
            df_transformed = df_transformed.join(df[non_media_factors])

        model = sm.OLS(
            y,
            sm.add_constant(df_transformed)
        ).fit()

        return model.ssr

    res = minimize(
        rss,
        x0=params_init,  # always list of initial guess, not dict or series or else
        args=(y, df, media_factors, is_adstock, is_saturate, saturate_function),
        bounds=bounds_init,
        tol=tol,
        method=optimization_method,
        options={
            'eps': np.array(
                [epsilon_step, epsilon_step, epsilon_step, epsilon_step] * len(media_factors)
                ),
            'iprint': 0, 
        }
    )
    return res.x


def show_comparison(media_params_optimal: Dict[str, Dict[str, float]], 
                    media_params_init: Dict[str, float]) -> None:
    """
    Show comparison between optimal and initial media parameters.

    Parameters:
    - media_params_optimal (Dict[str, Dict[str, float]]): 
        Dictionary containing optimal media parameters.
    - media_params_init (Dict[str, float]): 
        Dictionary containing initial media parameters.
    """
    print('Optimal: \n',
        pd.DataFrame(media_params_optimal))

    print('\nInitial: \n',
        pd.DataFrame(
            create_media_params_dict(
                media_params_init.get('params'),
                media_params_init.get('media'))
                ))


def plot_transformed_data(df: pd.DataFrame, df_transformed: pd.DataFrame,
                          media_factors: List[str], kpi_name: str = None,
                          ncols: int = 3) -> None:
    '''
    Compare transformed media variables and initial.

    Parameters:
    - df (pd.DataFrame): Data Frame before transformation. Usually, scaled.
    - df_transformed (pd.DataFrame): Data Frame after transformation
    - media_factors (List[str]): Media factors that were transformed. Exists in both data frames.
    - ncols (int): Number of columns at plot.

    Returns:
    - None: Just plot the data in output of Jupyter Notebook.
    '''
    nrows = len(media_factors) // ncols + (len(media_factors) % ncols > 0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, nrows * 5)
    )
    fig.subplots_adjust(wspace=5, hspace=5)
    for i, media_factor in enumerate(media_factors):
        row = i // ncols
        col = i % ncols

        if nrows == 1:
            ax = axes[col]
        elif nrows > 1:
            ax = axes[row, col]
        
        ax.plot(df[media_factor], label='Before transf',color='red')
        ax.plot(df_transformed[media_factor], label='After transf',color='green')

        if kpi_name is not None:
            ax.plot(df[kpi_name], label=f'KPI {kpi_name}')

        # ax.set_xlabel('Date')
        ax.set_ylabel(media_factor)
        ax.set_title(f'{media_factor}')
        ax.set_xticks([])
        ax.grid(True)
        ax.set_xticklabels([])
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_saturation_curves(media_params_optimal: Dict,
                           media_params_init: Dict,
                           saturate_func: str = 'saturate_hill',
                           x_steps: int = 1_000, x_max: float = 2.0,
                           y_max: float = 1.5, ncols: int = 3) -> None:
    '''
    Compare media factors dataframe before and after transforamtion.

    Parameters:
    - media_params_optimal (Dict): Optimal parameters for adstock and saturation transformation.
    - media_params_init (Dict): Initial parameters for adstock and saturation transformation.
    - saturate_func (str): Name of saturation function. Default is Hill funcstion.
    - x_max (int): Maximum of points on X-axis.
    - ncols (int): Number of columns at plot.

    Returns:
    - None: Just plot the data in output of Jupyter Notebook.
    '''
    if saturate_func == 'saturate_hill':
        saturate_function = saturate_hill
    elif saturate_func == 'saturate_power':
        saturate_function = saturate_power

    media_factors = list(media_params_optimal.keys())
    nrows = len(media_factors) // ncols + (len(media_factors) % ncols > 0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, nrows * 5)
    )
    for index, media_factor in enumerate(media_factors):
        row = index // ncols
        col = index % ncols
        if nrows == 1:
            ax = axes[col]
        elif nrows > 1:
            ax = axes[row, col]

        ax.scatter(
            (np.linspace(0, x_max, x_steps)),
            (pd.Series(
                saturate_function(
                    pd.Series(np.linspace(0, x_max, x_steps)),
                    media_params_init.get(media_factor).get('alpha'),
                    media_params_init.get(media_factor).get('beta'),
                    media_params_init.get(media_factor).get('gamma')
                )
            )),
            s=2,
            color='#C2C0A6',
            label='Initial'
        )

        ax.scatter(
            (np.linspace(0, x_max, x_steps)),
            (pd.Series(
                saturate_function(
                    pd.Series(np.linspace(0, x_max, x_steps)),
                    media_params_optimal.get(media_factor).get('alpha'),
                    media_params_optimal.get(media_factor).get('beta'),
                    media_params_optimal.get(media_factor).get('gamma')
                )
            )),
            s=2,
            color='#49F278',
            label='Optimal'
        )

        ax.set_title(f'Saturation of {media_factor} by func {saturate_func}')
        ax.set_xlabel('X value')
        ax.set_ylabel('Response of KPI')
        ax.set_ylim([0, y_max])
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_saturation_points(media_factors: List[str], 
                           media_params_optimal: Dict,
                           df: pd.DataFrame,
                           saturate_func: [str] = 'saturate_hill',
                           media_params_init: Dict = None,
                           is_adstocked: bool = False,
                           x_steps: int = 10_000,
                           y_max: float = 0.0,
                           ncols: int = 3
                           ) -> None:
    '''
    Plotting saturation optimized function and show points of real data on curves.

    Parameters:
    - media_factors (List[str]): List of media factors.
    - media_params_optimal (Dict): Optimizied parameters for adstock and saturation transformation.
    - df (pd.DataFrame): Data for plotting dots from reality on curves. Data is scaled.
    - saturate_func (str): Name of saturation function. Default is Hill function.
    - media_params_init (Dict): Parameters for adstock and saturation transformation before optimization.
    - is_adstocked (bool): Flag, if apply adstock transformation upon data from Data Frame.
    - x_steps (int): Number of points on X-axis.
    - y_max (float): Maximum value of Y-axis.
    - ncols (int): Number of columns at plot.
    
    Returns:
    - None: Just plot the data in output of Jupyter Notebook.
    '''
    if saturate_func == 'saturate_power':
        saturate_function = saturate_power
        find_saturated_point = find_saturated_point_power

    elif saturate_func == 'saturate_hill':
        saturate_function = saturate_hill
        find_saturated_point = find_saturated_point_hill
    
    nrows = len(media_factors) // ncols + (len(media_factors) % ncols > 0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, nrows * 5)
    )

    for index, media_factor in enumerate(media_factors):
        row = index // ncols
        col = index % ncols
        if nrows == 1:
            ax = axes[col]
        elif nrows > 1:
            ax = axes[row, col]

        median_point_adstocked = adstock(
                df[media_factor], media_params_optimal.get(media_factor).get('decay')
            ).replace(0, np.nan).median()
        median_point_not_adstocked = df[media_factor].replace(0, np.nan).median()

        max_point_adstocked = adstock(
                df[media_factor], media_params_optimal.get(media_factor).get('decay')
            ).replace(0, np.nan).max()
        max_point_not_adstocked = df[media_factor].replace(0, np.nan).max()

        if is_adstocked:
            median_point_plot = median_point_adstocked
            max_point_plot = max_point_adstocked
        else:
            median_point_plot = median_point_not_adstocked
            max_point_plot = max_point_not_adstocked

        median_point_saturation_effect = find_saturated_point(
            median_point_plot,
            media_params_optimal.get(media_factor).get('alpha'),
            media_params_optimal.get(media_factor).get('beta'),
            media_params_optimal.get(media_factor).get('gamma')
        )

        max_point_saturation_effect = find_saturated_point(
            max_point_plot,
            media_params_optimal.get(media_factor).get('alpha'),
            media_params_optimal.get(media_factor).get('beta'),
            media_params_optimal.get(media_factor).get('gamma')
        )

        # curve with optimal params
        ax.scatter(
            (np.linspace(0, np.ceil(max_point_plot), x_steps)),
            (pd.Series(
                saturate_function(
                    pd.Series(np.linspace(0, np.ceil(max_point_plot), x_steps)),
                    media_params_optimal.get(media_factor).get('alpha'),
                    media_params_optimal.get(media_factor).get('beta'),
                    media_params_optimal.get(media_factor).get('gamma')
                )
            )),
            s=3,
            color='#49F278',
            label='Saturation with optimal params'
        )

        # curve with default params if necessary
        if media_params_init is not None:
            ax.scatter(
                (np.linspace(0, np.ceil(max_point_plot), x_steps)),                
                (pd.Series(
                    saturate_function(
                        pd.Series(np.linspace(0, np.ceil(max_point_plot), x_steps)),
                        media_params_init.get(media_factor).get('alpha'),
                        media_params_init.get(media_factor).get('beta'),
                        media_params_init.get(media_factor).get('gamma')
                    )
                )),
                s=3,
                color='#C2C0A6',
                label='Saturation with initial params'
            )

        ax.scatter(
            (median_point_plot),
            (pd.Series(median_point_saturation_effect)),
            color='#F58C66',
            label=f'Median ({median_point_plot:.1f}, {median_point_saturation_effect:.1f})',
            s=100
        )

        ax.scatter(
            (max_point_plot),
            (pd.Series(max_point_saturation_effect)),
            color='#9D49F2',
            label=f'Historical maximum ({max_point_plot:.1f}, {max_point_saturation_effect:.1f})',
            s=100
        )

        ax.set_title(f'Saturation of {media_factor} by func {saturate_func}', fontsize=10)
        ax.set_xlabel(f'{media_factor} spends (scaled), is adstocked: {is_adstocked}')
        ax.set_ylabel('Media investments effect')
        
        if y_max > 0.0:  
            ax.set_ylim([0, y_max])
        ax.legend()

    plt.tight_layout()
    plt.show()


def optimize_params_by_methods(
        y: pd.Series,
        df: pd.DataFrame,
        media_factors: List[str],
        params_init: List[float],
        bounds_init: List[Tuple[float, float]],
        epsilon_step: float = 1e-08, 
        is_adstock: bool = True, 
        is_saturate: bool = True,
        saturate_function: str = 'saturate_power',
        tol: float = 1e-08,
        non_media_factors: List[str] = [],
        methods_list: List[str] = []) -> Dict:
    '''
    do

    Parameters:
    - 
    - media_factors (List[str]): Names of optimization algorythms in Scipy ('Powell', 'L-BFGS-B', 'CG', 'TNC', 'COBYLA', 'BFGS'): 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html 
    

    Returns:
    - Dict: Results in dict, key is the name of alorythm, value is media params for transformation.
    
    '''
    methods_result = {}

    media_params_init = create_media_params_dict(
        params_init,
        media_factors
    )

    for method in methods_list:
        media_params_optimal = optimize_media_params(
            y,
            df,
            media_factors,
            params_init,
            bounds_init,
            epsilon_step,
            is_adstock,
            is_saturate,
            saturate_function,
            tol,
            non_media_factors,
            method
        )

        methods_result[method] = create_media_params_dict(
            media_params_optimal,
            media_factors
        )

    return methods_result


def plot_saturation_methods(
        media_factors: List[str],
        params_by_methods: Dict,
        df: pd.DataFrame,
        saturate_func: [str] = 'saturate_power',
        exclude_methods: List[str] = [],
        media_params_init: Dict = None,
        is_adstocked: bool = False,
        x_steps: int = 10_000,
        y_max: float = 0.0,
        ncols: int = 3
        ) -> None:
    '''
    
    Plot
    '''
    if saturate_func == 'saturate_power':
        saturate_function = saturate_power
        find_saturated_point = find_saturated_point_power

    elif saturate_func == 'saturate_hill':
        saturate_function = saturate_hill
        find_saturated_point = find_saturated_point_hill
    
    nrows = len(media_factors) // ncols + (len(media_factors) % ncols > 0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, nrows * 5)
    )

    for index, media_factor in enumerate(media_factors):
        row = index // ncols
        col = index % ncols
        if nrows == 1:
            ax = axes[col]
        elif nrows > 1:
            ax = axes[row, col]

        x_max = 0.0

        methods_to_plot = list(set(params_by_methods.keys()) - set(exclude_methods))

        for method in methods_to_plot:

            median_point_adstocked = adstock(
                    df[media_factor], params_by_methods.get(method).get(media_factor).get('decay')
                ).replace(0, np.nan).median()
            
            median_point_not_adstocked = df[media_factor].replace(0, np.nan).median()

            max_point_adstocked = adstock(
                    df[media_factor], params_by_methods.get(method).get(media_factor).get('decay')
                ).replace(0, np.nan).max()
            max_point_not_adstocked = df[media_factor].replace(0, np.nan).max()

            if is_adstocked:
                median_point_plot = median_point_adstocked
                max_point_plot = max_point_adstocked
            else:
                median_point_plot = median_point_not_adstocked
                max_point_plot = max_point_not_adstocked

            if max_point_plot > x_max:
                x_max = max_point_plot

            median_point_saturation_effect = find_saturated_point(
                median_point_plot,
                params_by_methods.get(method).get(media_factor).get('alpha'),
                params_by_methods.get(method).get(media_factor).get('beta'),
                params_by_methods.get(method).get(media_factor).get('gamma')
            )

            max_point_saturation_effect = find_saturated_point(
                max_point_plot,
                params_by_methods.get(method).get(media_factor).get('alpha'),
                params_by_methods.get(method).get(media_factor).get('beta'),
                params_by_methods.get(method).get(media_factor).get('gamma')
            )

            ax.scatter(
                (np.linspace(0, np.ceil(x_max), x_steps)),
                (pd.Series(
                    saturate_function(
                        pd.Series(np.linspace(0, np.ceil(x_max), x_steps)),
                        params_by_methods.get(method).get(media_factor).get('alpha'),
                        params_by_methods.get(method).get(media_factor).get('beta'),
                        params_by_methods.get(method).get(media_factor).get('gamma')
                    )
                )),
                s=2,
                label=f'Algo: {method}'
            )

            ax.scatter(
                (median_point_plot),
                (pd.Series(median_point_saturation_effect)),
                color='black',
            )

            ax.scatter(
                (max_point_plot),
                (pd.Series(max_point_saturation_effect)),
                color='red',
            )

        # curve with initial params saturation
        if media_params_init is not None:
            ax.scatter(
                (np.linspace(0, np.ceil(x_max), x_steps)),                
                (pd.Series(
                    saturate_function(
                        pd.Series(np.linspace(0, np.ceil(x_max), x_steps)),
                        media_params_init.get(media_factor).get('alpha'),
                        media_params_init.get(media_factor).get('beta'),
                        media_params_init.get(media_factor).get('gamma')
                    )
                )),
                s=1,
                color='#C2C0A6',
                label='Saturation with initial params'
            )            

        ax.set_title(f'Saturation of {media_factor} by func {saturate_func}', fontsize=10)
        ax.set_xlabel(f'{media_factor} spends (scaled), is adstocked: {is_adstocked}')
        ax.set_ylabel('Media investments effect')

        if y_max > 0.0:
            ax.set_ylim([0, y_max])
        ax.legend()

    plt.tight_layout()
    plt.show()


# grid-search
def calculate_media_params_grid_search(df: pd.DataFrame, media_factors: List[str],
                                      kpi_name: str, media_params_input: pd.DataFrame,
                                      step: float = 0.1, non_media_factors: List[str] = [],
                                      adstock_func: str = None,
                                      saturate_func: str = 'saturate_power',
                                      n_digits: int = 3, is_log : bool = False) -> pd.DataFrame:
    '''
    Independent grid-search optimization of parameters for media transformation. Factor is not changed after params optimization.
    Each row is independent calculation. Selection after could be done by lowest rss per specific media. Model quality per 
    media is 'change one media - other media and nonmedia factors are constant.
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    df_x = df[list(set(media_factors).union(set(non_media_factors)))].copy()
    
    if saturate_func == 'saturate_power':
        saturation_function = saturate_power
    elif saturate_func == 'saturate_hill':
        saturation_function = saturate_hill

    params_df = dict()
    for media_factor in tqdm(media_factors):

        logging.info(f'Starting media: {media_factor}')

        params_grid = [
            np.arange(
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'decay_bound_min']),
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'decay_bound_max'])  + step,
                step
            ),
            np.arange(
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'alpha_bound_min']),
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'alpha_bound_max'])  + step, 
                step
            ),
            np.arange(
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'beta_bound_min']),
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'beta_bound_max'])  + step,
                step,
            ),
            np.arange(
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'gamma_bound_min']),
                float(media_params_input.loc[media_params_input['media'] == media_factor, 'gamma_bound_max'])  + step,
                step
            )
        ]

        params_combinations = list(itertools.product(*params_grid))
        params_combinations_df = pd.DataFrame(
            params_combinations,
            columns=[
                'decay',
                'alpha',
                'beta',
                'gamma',
            ])

        for idx in params_combinations_df.index:

            media_adstocked = adstock(
                df_x[media_factor],
                params_combinations_df.iloc[idx]['decay']
            )

            media_adstocked_sat = saturation_function(
                media_adstocked,
                params_combinations_df.iloc[idx]['alpha'],
                params_combinations_df.iloc[idx]['beta'],
                params_combinations_df.iloc[idx]['gamma']
            )
            
            media_adstocked_sat_df = pd.DataFrame(
                data=media_adstocked_sat,
                index=df_x.index,
                columns=[media_factor]
            )

            x_temp = df_x.drop([media_factor], axis=1).join(media_adstocked_sat_df)

            model_factors = list(set(media_factors).union(set(non_media_factors)))

            model_ols = sm.OLS(
                df[kpi_name],
                sm.add_constant(x_temp[model_factors])
            ).fit()
            params_combinations_df.at[idx, 'rss'] = model_ols.ssr
            params_combinations_df.at[idx, 'model_number'] = idx
            params_combinations_df.at[idx, 'media'] = media_factor
            params_combinations_df.at[idx, 'r2_adj'] = np.round(model_ols.rsquared_adj, n_digits)
            params_combinations_df.at[idx, 'r2'] = np.round(model_ols.rsquared, n_digits)
            params_combinations_df.at[idx, 'dw'] = np.round(sm.stats.stattools.durbin_watson(model_ols.resid, axis=0), n_digits)
            params_combinations_df.at[idx, 'mape'] = np.round(mean_absolute_percentage_error(model_ols.fittedvalues.values, df[kpi_name]), n_digits)
            params_combinations_df.at[idx, 'f_model'] = np.round(model_ols.fvalue, n_digits)
            params_combinations_df.at[idx, 'f_model_pvalue'] = np.round(model_ols.f_pvalue, n_digits)
            params_combinations_df.at[idx, 'aic'] = np.round(model_ols.aic, n_digits)
            params_combinations_df.at[idx, 'bic'] = np.round(model_ols.bic, n_digits)
            params_combinations_df.at[idx, 'model'] = model_ols
            params_combinations_df.at[idx, 'coeff_corr'] = np.corrcoef(df[kpi_name].values, media_adstocked_sat_df[media_factor].values)[0][1]
            params_combinations_df.at[idx, 'adstock_func'] = adstock_func
            params_combinations_df.at[idx, 'saturate_func'] = saturate_func

            # logging.info(f'{params_combinations_df.iloc[idx]}')
            
            # ToDo
            # params_combinations_df.to_csv(f'../log/log{}.csv')

        params_df[media_factor] = params_combinations_df

    params_dataframes = []
    for media_param in params_df.keys():
        params_dataframes.append(params_df[media_param])

    params_df = pd.concat(params_dataframes, ignore_index=True)
    return params_df


def select_best_params_grid_search(params_df, media_factors, criteria: str = 'rss') -> Dict:
    '''
    '''
    best_params = dict()
    for media_factor in media_factors:

        if criteria in ['rss']:
            best_model_media = params_df.loc[params_df['media'] == media_factor].sort_values(by=criteria, ascending=True)[:1]
        else:
            best_model_media = params_df.loc[params_df['media'] == media_factor].sort_values(by=criteria, ascending=False)[:1]
        best_params[media_factor] = dict()
        best_params[media_factor]['decay'] = best_model_media['decay'].values[0]
        best_params[media_factor]['alpha'] = best_model_media['alpha'].values[0]
        best_params[media_factor]['beta'] = best_model_media['beta'].values[0]
        best_params[media_factor]['gamma'] = best_model_media['gamma'].values[0]
    return best_params


def calculate_media_params_grid_search_conseq(
        df: pd.DataFrame, media_factors: List[str],
        kpi_name: str, media_params_input: pd.DataFrame,
        step: float = 0.1, is_transform_df: bool = True,
        non_media_factors: List[str] = [],
        adstock_func: str = None,
        saturate_func: str = 'saturate_power',
        criteria: str = 'rss',
        n_digits: int = 3) -> pd.DataFrame:
    '''
    Consequent grid-search optimization for adstock (decay) and saturation functions (with 3 params) (Power, Hill, etc) 
    with step-by-step changing transformation of media variables. It works if flag `is_transform_df` is True.
    If it False function works like `calculate_media_params_grid_search`, but much slower.
    Order of including media factors into optimization is important, as each next media factor is optimizied on the data frame, 
    where previuos medua factor has been transformed.


    Parameters:
    - is_transform_df (bool): Whether transform variable in dataset after params optimization. Default is not transform.
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df_x = df[list(set(media_factors).union(set(non_media_factors)))].copy()

    # saturate functions
    if saturate_func == 'saturate_power':
        saturation_function = saturate_power
    elif saturate_func == 'saturate_hill':
        saturation_function = saturate_hill
    
    # make combinations of media 
    media_combinations = list(itertools.permutations(media_factors, len(media_factors)))
    media_combinations = [list(elem) for elem in media_combinations]

    # make global df
    global_columns = [
        'media_combination',
        'media',
        'decay',
        'alpha',
        'beta',
        'gamma',
        'rss',
        'r2_adj',
        'r2',
        'dw',
        'mape',
        'f_model',
        'f_model_pvalue',
        'aic',
        'bic',
        'coeff_corr',
        'adstock_func',
        'saturate_func',
    ]
    global_results = pd.DataFrame(
        columns=global_columns
    )

    # iterate via combinations
    for media_combination in tqdm(media_combinations):

        logging.info(f'Starting media_combination {media_combination}')

        for idx_combination, media_factor in enumerate(tqdm(media_combination)):
            # logging.info(f'Starting params grid for media_combination {media_combination} for media {media_factor}')

            # optimize first media in combination
            params_grid = [
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'decay_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'decay_bound_max']) + step,
                    step
                ),
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'alpha_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'alpha_bound_max']) + step,
                    step
                ),
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'beta_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'beta_bound_max']) + step,
                    step,
                ),
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'gamma_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'gamma_bound_max']) + step,
                    step
                )
            ]

            params_combinations = list(itertools.product(*params_grid))
            params_combinations_df = pd.DataFrame(
                params_combinations,
                columns=[
                    'decay',
                    'alpha',
                    'beta',
                    'gamma',
                ])

            for idx in params_combinations_df.index:
                media_adstocked = adstock(
                    df_x[media_factor],
                    params_combinations_df.iloc[idx]['decay']
                )

                media_adstocked_sat = saturation_function(
                    media_adstocked,
                    params_combinations_df.iloc[idx]['alpha'],
                    params_combinations_df.iloc[idx]['beta'],
                    params_combinations_df.iloc[idx]['gamma']
                )
                
                media_adstocked_sat_df = pd.DataFrame(
                    data=media_adstocked_sat,
                    index=df_x.index,
                    columns=[media_factor]
                )

                x_temp = df_x.drop([media_factor], axis=1).join(media_adstocked_sat_df)
                model_factors = list(set(media_factors).union(set(non_media_factors)))
                model_ols = sm.OLS(
                    df[kpi_name],
                    sm.add_constant(x_temp[model_factors])
                ).fit()

                params_combinations_df.at[idx, 'media'] = media_factor
                params_combinations_df.at[idx, 'rss'] = model_ols.ssr
                params_combinations_df.at[idx, 'r2_adj'] = np.round(model_ols.rsquared_adj, n_digits)
                params_combinations_df.at[idx, 'r2'] = np.round(model_ols.rsquared, n_digits)
                params_combinations_df.at[idx, 'dw'] = np.round(sm.stats.stattools.durbin_watson(model_ols.resid, axis=0), n_digits)
                params_combinations_df.at[idx, 'mape'] = np.round(mean_absolute_percentage_error(model_ols.fittedvalues.values, df[kpi_name]), n_digits)
                params_combinations_df.at[idx, 'f_model'] = np.round(model_ols.fvalue, n_digits)
                params_combinations_df.at[idx, 'f_model_pvalue'] = np.round(model_ols.f_pvalue, n_digits)
                params_combinations_df.at[idx, 'aic'] = np.round(model_ols.aic, n_digits)
                params_combinations_df.at[idx, 'bic'] = np.round(model_ols.bic, n_digits)
                params_combinations_df.at[idx, 'media_combination'] = '->'.join(media_combination)
                params_combinations_df.at[idx, 'coeff_corr'] = np.corrcoef(df[kpi_name].values, media_adstocked_sat_df[media_factor].values)[0][1]
                params_combinations_df.at[idx, 'adstock_func'] = adstock_func
                params_combinations_df.at[idx, 'saturate_func'] = saturate_func
            # write best in best params df with other metrics on model in local df
            # best_params_per_media = params_combinations_df.sort_values(by=criteria, ascending=True)[:1]

            if criteria in ['rss']:
                best_params_per_media = params_combinations_df.sort_values(by=criteria, ascending=True)[:1]
            else:
                best_params_per_media = params_combinations_df.sort_values(by=criteria, ascending=False)[:1]
            
            # if coeff corr - chamge to ascending False 

            global_results = pd.concat([global_results, best_params_per_media], axis=0, ignore_index=True)

            # logging.info(f'best params for media: {media_factor} decay: {best_params_per_media['decay'].values[0]} ' 
            #              f'alpha: {best_params_per_media['alpha'].values[0]} beta: {best_params_per_media['beta'].values[0]} gamma: {best_params_per_media['gamma'].values[0]} rss:{best_params_per_media['rss'].values[0]:.3f}')

            if is_transform_df:
                # transform media factor by best params
                media_adstocked_best = adstock(
                    df_x[media_factor],
                    best_params_per_media['decay'].values[0]
                )

                media_adstocked_sat = saturation_function(
                        media_adstocked_best,
                        best_params_per_media['alpha'].values[0],
                        best_params_per_media['beta'].values[0],
                        best_params_per_media['gamma'].values[0]
                    )

                df_x[media_factor] = media_adstocked_sat

            if idx_combination == len(media_factors) - 1:
                best_params_per_combination = best_params_per_media.copy()
                best_params_per_combination['media'] = 'Total'
                global_results = pd.concat([global_results, best_params_per_combination], axis=0, ignore_index=True)

            logging.info(f'Finished media {media_factor}')

    logging.disable(logging.CRITICAL)
    return global_results


def select_best_params_grid_search_conseq(params_df_conseq, media_factors, criteria: str = 'rss') -> Dict:
    '''
    '''
    if criteria in ['rss']:
        best_combination_name = params_df_conseq.loc[params_df_conseq['media'] == 'Total'].sort_values(by=criteria, ascending=True)[:1]['media_combination'].values[0]
    else:
        best_combination_name = params_df_conseq.loc[params_df_conseq['media'] == 'Total'].sort_values(by=criteria, ascending=False)[:1]['media_combination'].values[0]

    best_combination = params_df_conseq[params_df_conseq['media_combination'] == best_combination_name]
    best_params = dict()
    best_params['media_combination'] = best_combination_name
    for media_factor in media_factors:
        best_params[media_factor] = dict()
        best_params[media_factor]['decay'] = best_combination[best_combination['media'] == media_factor]['decay'].values[0]
        best_params[media_factor]['alpha'] = best_combination[best_combination['media'] == media_factor]['alpha'].values[0]
        best_params[media_factor]['beta'] = best_combination[best_combination['media'] == media_factor]['beta'].values[0]
        best_params[media_factor]['gamma'] = best_combination[best_combination['media'] == media_factor]['gamma'].values[0]
    return best_params


def prepare_media_params_init_lt(df: pd.DataFrame) -> dict:
    """
    Return dictionary of media parameters initialization values for shape and scale.

    Args:
        df (pd.DataFrame): DataFrame containing media parameters.

    Returns:
        dict: Dictionary containing initialization values for media parameters.
    """
    params_names = [
        'shape_param',
        'scale_param',
        'st_period_param',
    ]

    initializiation_values = dict()

    initializiation_values['media'] = list(df['media'])

    params = np.array([])
    for param in params_names:
        params = np.concatenate(
            (
                params,
                df[param].to_numpy()
            )
        )
    initializiation_values['params'] = list(params)

    shape_t = tuple(
        zip(
            df['shape_bound_min'].to_numpy(),
            df['shape_bound_max'].to_numpy(),
        )
    )
    
    scale_t = tuple(
        zip(
            df['scale_bound_min'].to_numpy(),
            df['scale_bound_max'].to_numpy(),
        )
    )

    st_period_t = tuple(
        zip(
            df['st_period_bound_min'].to_numpy(),
            df['st_period_bound_max'].to_numpy(),
        )
    )

    bounds_list = [shape_t, scale_t, st_period_t]

    bounds_temp = list()
    for bounds in bounds_list:
        bounds_temp.extend(list(bounds))
    bounds_result = tuple(bounds_temp)
    
    initializiation_values['bounds'] = bounds_result

    return initializiation_values


def create_media_params_dict_lt(params: pd.DataFrame, media_factors: List[str]):
    """
    Create a dictionary of media parameters.

    Parameters:
    - params (list): List of parameter values.
    - media_factors (list): List of media factors.

    Returns:
    - dict: Dictionary containing media parameters.
    """
    media_params = {}
    n = len(media_factors)
    for i, key in enumerate(media_factors):
        media_params[key] = {
            'shape': params[i::n][0],
            'scale': params[i::n][1],
            'st_period': params[i::n][2],
        }
    return media_params


def calculate_lt_weights(size: int, shape: float, scale: float, st_period: int):
    exp_ = []
    for i in range(size + 1):
        exp_.append(1 / (1 + np.exp(scale * i - shape)))
    return pd.Series(exp_)


def adstock_lt_effect(series: pd.Series, shape: float, scale: float, st_period: int) -> pd.Series:
    '''
    Calculate Long Term effect of flight.

    Parameters:
    - shape (float): number of points where effect is sees,  larger shape means "Fat" long-term effect
    - scale (float): speed of slowing , larger scale means shorter delayed effect, lower scale means  Longer effect
    
    Returns:
    - pd.Series: Transformed data series.
    '''
    if len(series) == 0:
        return 0.
    else:
        x_series = series.copy()
        exp_ = []
        for i in range(len(x_series) + 1):
            exp_.append(1 / (1 + np.exp(scale * i - shape)))

        # adstock data
        adstocked_ad = pd.DataFrame(index=x_series.index)
        x_series.fillna(0, inplace=True)
        x_series = x_series.reset_index(drop=True)

        for week, x_week in enumerate(x_series.values):
            if week + int(st_period) + 1 >= len(x_series):
                pass
            else:
                adstocked_ad[str(week)] = 0.
                adstocked_ad[str(week)][week + int(st_period) + 1:] = [i * x_week for i in exp_[1: -(week + int(st_period) + 1)]]

        return adstocked_ad.sum(axis=1)


def calculate_media_params_grid_search_conseq_lt(
        df: pd.DataFrame, media_factors: List[str],
        kpi_name: str, media_params_input: pd.DataFrame,
        step_shape: float = 1.,
        step_scale: float = 0.2,
        step_st_period: int = 1,
        is_transform_df: bool = True,
        non_media_factors: List[str] = [],
        adstock_func: str = 'adstock_lt_effect',
        saturate_func: str = None,
        criteria: str = 'rss',
        n_digits: int = 3) -> pd.DataFrame:
    '''
    Consequent grid-search optimization parameters for long-term adstock transformation (shape, scale, st_period).
    Step-by-step changing transformation of media variables is included: it works if flag `is_transform_df` is True.
    If it False function media factor in data frame after optimization is not transformed.
    Order of including media factors into optimization is important, as each next media factor is optimizied on the data frame, 
    where previuos media factor has been transformed.

    Parameters:
    - df (pd.DataFrame)
    - media_factors
    - shape (float): Number of points where effect is seen, larger shape means "Fat" long-term effect.
    - scale (float): Speed of slowing, larger scale means shorter delayed effect, lower scale means longer effect.
    - step_st_period (int): Window for weighting points.
    - is_transform_df (bool): Whether transform variable in dataset after params optimization. Default is `True` (== transform).
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df_x = df[list(set(media_factors).union(set(non_media_factors)))].copy()

    # adstock long-term functions
    if adstock_func == 'adstock_lt_effect':
        adstock_function = adstock_lt_effect
    
    # make combinations of media
    # if is_transform_df os False it means that combinations would be enough, and permutations are exhaustive
    if is_transform_df:
        media_combinations = list(itertools.permutations(media_factors, len(media_factors)))
    else:
        media_combinations = list(itertools.combinations(media_factors, len(media_factors)))

    media_combinations = [list(elem) for elem in media_combinations]

    # make global df
    global_columns = [
        'media_combination',
        'media',
        'shape',
        'scale',
        'st_period',
        'rss',
        'r2_adj',
        'r2',
        'dw',
        'mape',
        'f_model',
        'f_model_pvalue',
        'aic',
        'bic',
        'corr_coeff',
        'adstock_func',
        'saturate_func',
    ]
    global_results = pd.DataFrame(
        columns=global_columns
    )

    # iterate via combinations
    for media_combination in tqdm(media_combinations):

        logging.info(f'Starting media_combination {media_combination}')
 
        for idx_combination, media_factor in enumerate(tqdm(media_combination)):

            # logging.info(f'Starting params grid for media_combination {media_combination} for media {media_factor}')

            params_grid = [
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'shape_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'shape_bound_max']) + step_shape,
                    step_shape
                ),
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'scale_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'scale_bound_max']) + step_scale, 
                    step_scale
                ),
                np.arange(
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'st_period_bound_min']),
                    float(media_params_input.loc[media_params_input['media'] == media_factor, 'st_period_bound_max']) + step_st_period, 
                    step_st_period
                ),
            ]

            params_combinations = list(itertools.product(*params_grid))
            params_combinations_df = pd.DataFrame(
                params_combinations,
                columns=[
                    'shape',
                    'scale',
                    'st_period',
                ])

            for idx in params_combinations_df.index:

                media_adstocked_lt = adstock_function(
                    df_x[media_factor],
                    params_combinations_df.iloc[idx]['shape'],
                    params_combinations_df.iloc[idx]['scale'],
                    params_combinations_df.iloc[idx]['st_period'],
                )

                media_adstocked_df = pd.DataFrame(
                    data=media_adstocked_lt,
                    index=df_x.index,
                    columns=[media_factor]
                )

                x_temp = df_x.drop([media_factor], axis=1).join(media_adstocked_df)
                model_factors = list(set(media_factors).union(set(non_media_factors)))

                model_ols = sm.OLS(
                    df[kpi_name],
                    sm.add_constant(x_temp[model_factors])
                ).fit()

                params_combinations_df.at[idx, 'media'] = media_factor
                params_combinations_df.at[idx, 'rss'] = model_ols.ssr
                params_combinations_df.at[idx, 'r2_adj'] = np.round(model_ols.rsquared_adj, n_digits)
                params_combinations_df.at[idx, 'r2'] = np.round(model_ols.rsquared, n_digits)
                params_combinations_df.at[idx, 'dw'] = np.round(sm.stats.stattools.durbin_watson(model_ols.resid, axis=0), n_digits)
                params_combinations_df.at[idx, 'mape'] = np.round(mean_absolute_percentage_error(model_ols.fittedvalues.values, df[kpi_name]), n_digits)
                params_combinations_df.at[idx, 'f_model'] = np.round(model_ols.fvalue, n_digits)
                params_combinations_df.at[idx, 'f_model_pvalue'] = np.round(model_ols.f_pvalue, n_digits)
                params_combinations_df.at[idx, 'aic'] = np.round(model_ols.aic, n_digits)
                params_combinations_df.at[idx, 'bic'] = np.round(model_ols.bic, n_digits)
                params_combinations_df.at[idx, 'media_combination'] = '->'.join(media_combination)
                params_combinations_df.at[idx, 'corr_coeff'] = np.corrcoef(df[kpi_name].values, media_adstocked_df[media_factor].values)[0][1]
                params_combinations_df.at[idx, 'adstock_func'] = adstock_func
                params_combinations_df.at[idx, 'adstock_func'] = saturate_func

            # write best in best params df with other metrics on model in local df
            if criteria in ['rss']:
                best_params_per_media = params_combinations_df.sort_values(by=criteria, ascending=True)[:1]
            else:
                best_params_per_media = params_combinations_df.sort_values(by=criteria, ascending=False)[:1]
            
            global_results = pd.concat([global_results, best_params_per_media], axis=0, ignore_index=True)

            # logging.info(f'best params for media: {media_factor} shape: {best_params_per_media['shape'].values[0]}'
            #              f'scale: {best_params_per_media['scale'].values[0]}'
            #              f'rss:{best_params_per_media['rss'].values[0]:.3f}')
            
            if is_transform_df:
                # transform media factor by best params
                media_adstocked_lt = adstock_function(
                    df_x[media_factor],
                    best_params_per_media['shape'].values[0],
                    best_params_per_media['scale'].values[0],
                    best_params_per_media['st_period'].values[0],
                )

                df_x[media_factor] = media_adstocked_lt

            if idx_combination == len(media_factors) - 1:
                best_params_per_combination = best_params_per_media.copy()
                best_params_per_combination['media'] = 'Total'
                global_results = pd.concat([global_results, best_params_per_combination], axis=0, ignore_index=True)

            # logging.info(f'Finished media {media_factor}')

    # logging.disable(logging.CRITICAL)
    return global_results


def select_best_params_grid_search_conseq_lt(params_df_conseq: pd.DataFrame, media_factors: List[str],
                                             criteria: str = 'rss') -> Dict:
    '''
    Utility for selecting best media params from data frame with params combinations, gotten after optimization.
    Used for long term optimization params (shape, scale, st_period).

    Parameters:
    - params_df_conseq (pd.DataFrame): Data Frame with quality metrics of different combinations of
        params for media transformation.
    - media_factors (List[str]): List of media factors for selection from Data Frame with params. 
    - criteria (str): Criteria for selecting best combination. Default id `rss`. If `rss`, select combination with 
        lowest rss value.

    Returns:
    - Dict: Media params for further usage in transformation functions. Long term optimization params (shape, scale, st_period).
    '''
    if criteria in ['rss']:
        best_combination_name = params_df_conseq.loc[params_df_conseq['media'] == 'Total'].sort_values(by=criteria, ascending=True)[:1]['media_combination'].values[0]
    else:
        best_combination_name = params_df_conseq.loc[params_df_conseq['media'] == 'Total'].sort_values(by=criteria, ascending=False)[:1]['media_combination'].values[0]

    best_combination = params_df_conseq[params_df_conseq['media_combination'] == best_combination_name]
    best_params = dict()
    best_params['media_combination'] = best_combination_name
    for media_factor in media_factors:
        best_params[media_factor] = dict()
        best_params[media_factor]['shape'] = best_combination[best_combination['media'] == media_factor]['shape'].values[0]
        best_params[media_factor]['scale'] = best_combination[best_combination['media'] == media_factor]['scale'].values[0]
        best_params[media_factor]['st_period'] = best_combination[best_combination['media'] == media_factor]['st_period'].values[0]

    return best_params


def transform_media_data_lt(df: pd.DataFrame, params: Dict[str, Dict[str, float]],
                            media_factors: List[str], is_transform_df: bool = False,
                            adstock_func: str = 'adstock_lt_effect') -> pd.DataFrame:
    '''
    Transform data frame using long-term adstock function. Mainly for marketing metrics modelling.

    Parameters:
    - df (pd.DataFrame): Data Frame with data.
    - params (Dict): Params with shape, scale, st_period for long-term effect calculation.
    - media_factors (List[str]): List of media factors for transformation.
    - is_transform (bool): Flag for defining, whether order of media factors 
        transformation should be taking into account (consequent) or not (independent). 
        Default is `False`.
    - adstock_func (str): Name of adstock transformation function. Default is `adstock_lt_effect`. 
    List of functions could be extended.

    Returns:
    - pd.DataFrame: Data Frame with transformed media factors.
    '''
    if adstock_func == 'adstock_lt_effect':
        adstock_function = adstock_lt_effect

    if is_transform_df:
        order = params['media_combination'].split('->')
    else:
        order = media_factors.copy()

    media_transformed = pd.DataFrame()

    for media_factor in order:
        media_transformed[media_factor] = adstock_function(
            df[media_factor],
            params.get(media_factor).get('shape'),
            params.get(media_factor).get('scale'),
            params.get(media_factor).get('st_period'),
        )

    return media_transformed



def create_optimal_params_dict(df_params, media_factors):
    '''
    Function from Islam as better version of create parasms dict.
    '''
    media_params = dict()
    df_params.set_index(['media'], inplace=True)
    cols = df_params.columns
    n = len(media_factors)
    for i, key in enumerate(media_factors):
        media_params[key] = dict()
        for column in cols:
            media_params[key][column] = df_params.loc[key][column]
    return media_params