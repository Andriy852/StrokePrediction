import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
import math
from sklearn.feature_selection import mutual_info_classif

def find_best_parameters(pipeline: Pipeline, 
                         param_grid: Dict[str, Any], 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         scoring: Dict[str, Union[str, Any]], 
                         cv: StratifiedKFold, 
                         refit: str) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Perform a grid search to find the best hyperparameters for a given pipeline.

    Parameters:
    -----------
    pipeline : Pipeline
        The machine learning pipeline containing the model and any preprocessing steps.
    
    param_grid : Dict[str, Any]
        The hyperparameters to be tuned, along with the range
        of values for each hyperparameter.
    
    X : pd.DataFrame
        The input features for training the model.
    
    y : pd.Series
        The target variable for training the model.
    
    scoring : Dict[str, Union[str, Any]]
        A dictionary containing the scoring metrics to evaluate the model's performance.
        The dictionary can have string keys associated with either string metrics 
        (like 'accuracy', 'recall') or custom scorer objects.
    
    cv : StratifiedKFold
        The cross-validation splitting strategy. 
        
    refit : str
        The scoring metric to use for refitting the best model after grid search.

    Return:
    best_model : Pipeline
        The pipeline with the best combination
        of hyperparameters found by the grid search.
    
    best_model_res : pd.DataFrame
        A DataFrame containing the average results of the best model
    """
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, cv=cv, scoring=scoring,
                               refit=refit, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_model_res = grid_search_results(grid_search, scoring, refit)
    return best_model, best_model_res

def plot_confusion_matrices(pipelines: List[BaseEstimator], X: np.ndarray,
                            y: np.ndarray,  figsize: Tuple[int, int]) -> None:
    """
    Plot confusion matrices for a list of models using cross-validated predictions.

    Parameters:
    models (List[BaseEstimator]): A list of scikit-learn estimator instances.
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.

    Returns:
    None: The function plots confusion matrices for the provided models.
    """
    fig = plt.figure(figsize=figsize)

    for i, pipe in enumerate(pipelines):
        ax = fig.add_subplot(math.ceil(len(pipelines) / 3), 3, i + 1)
        y_pred = cross_val_predict(pipe, X, y, cv=5)

        if type(pipe) == Pipeline:
            title = type(pipe[-1]).__name__
        else:
            title = type(pipe).__name__
        ax.set_title(title)

        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

def get_scores(model: BaseEstimator, X: np.ndarray,
               y: np.ndarray, fit: bool = True) -> Dict[str, float]:
    """
    Compute performance scores on the data.

    Parameters:
    model (BaseEstimator): The machine learning model
    X (np.ndarray): The feature matrix used.
    y (np.ndarray): The target vector used.
    fit (bool): If True, the model will be fitted to the data. Default is True.

    Returns:
    Dict[str, float]: A dictionary containing accuracy, recall, precision, and f1 scores.
    """
    if fit:
        model.fit(X, y)

    model_predict = model.predict(X)

    scores = {
        "accuracy": accuracy_score(y, model_predict),
        "recall": recall_score(y, model_predict),
        "precision": precision_score(y, model_predict),
        "f1": f1_score(y, model_predict)
    }

    return scores

def cross_val_scores(model: BaseEstimator, X: np.ndarray,
                     y: np.ndarray, scoring: List[str], cv: int) -> pd.DataFrame:
    """
    Perform cross-validation on the given model and compute specified scoring metrics.

    Parameters:
    model (BaseEstimator): The machine learning model to be evaluated.
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    scoring (List[str]): A list of scoring methods to evaluate the model.
    cv (int): The number of cross-validation folds.

    Returns:
    pd.DataFrame: A DataFrame containing the cross-validation scores for each metric.
    """
    results = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=False)

    scores = {key: results[key] for key in results if key.startswith('test_')}
    scores = {key[5:]: scores[key] for key in scores}  # Remove 'test_' prefix from keys

    scores_df = pd.DataFrame(scores)

    return scores_df

def grid_search_results(grid: GridSearchCV, scoring: List[str],
                        main_score: str) -> pd.DataFrame:
    """
    Extract the best scores from a GridSearchCV object for specified scoring metrics.

    Parameters:
    grid (GridSearchCV): The GridSearchCV object after fitting.
    scoring (List[str]): A list of scoring metrics used in GridSearchCV.
    main_score (str): The primary scoring metric used to identify the best model.

    Returns:
    pd.DataFrame: A DataFrame containing the mean and standard deviation of test scores
    for each metric at the best model's index.
    """
    results = grid.cv_results_
    best_accuracy_idx = np.argmax(results[f'mean_test_{main_score}'])

    # Create DataFrame to store mean and std of test scores for each metric
    scores = pd.DataFrame(index=scoring, columns=["mean", "std"])

    # Populate the DataFrame with the best scores
    for score in scoring:
        scores.loc[score, "mean"] = results[f'mean_test_{score}'][best_accuracy_idx]
        scores.loc[score, "std"] = results[f'std_test_{score}'][best_accuracy_idx]

    return scores

def customize_bar(position: str, axes, 
                  values_font=12, pct=False, round_to=0) -> None:
    """
    Function, which customizes bar chart
    Takes axes object and:
        - gets rid of spines
        - modifies ticks
        - adds value above each bar
    Parameters:
        - position(str): modify the bar depending on how the
        bars are positioned: vertically or horizontally
    Return: None
    """
    # get rid of spines
    for spine in axes.spines.values():
        spine.set_visible(False)
    # modify ticklabels
    if position == "v":
        axes.set_yticks([])
        for tick in axes.get_xticklabels():
            tick.set_rotation(0)
    if position == "h":
        axes.set_xticks([])
        for tick in axes.get_yticklabels():
            tick.set_rotation(0)
    # add height value above each bar
    for bar in axes.patches:
        if bar.get_width() == 0:
            continue
        if position == "v":
            text_location = (bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1/100*bar.get_height())
            value = bar.get_height()
            location = "center"
        elif position == "h":
            text_location = (bar.get_width(),
                             bar.get_y() + bar.get_height() / 2)
            value = bar.get_width()
            location = "left"
        if pct:
            value = f"{round(value * 100, round_to)}%"
        elif round_to == 0:
            value = str(int(value))
        else:
            value = str(round(value, round_to))
        axes.text(text_location[0],
                text_location[1],
                str(value),
                fontsize=values_font,
                ha=location)
        
def plot_cat_columns(data: pd.DataFrame, columns: List[str], 
                     title: str, figsize: Tuple[int, int]) -> None:
    """
    Plots count plots for specified categorical columns in a DataFrame.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    columns : List[str]
        The list of column names to be plotted.
    title : str
        The title for the entire figure.
    figsize : Tuple[int, int]
        The size of the figure (width, height).

    Returns:
       None
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.91)
    for i, column in enumerate(columns):
        ax = fig.add_subplot(3, 3, i+1)
        sns.countplot(x=column, data=data, 
                      ax=ax, color="red")

        customize_bar(axes=ax, position="v", values_font=10)
        ax.set_xlabel("")
        ax.set_title(column.capitalize(), fontsize=12)

        # get rid of ylabels for middle or right axes
        if i % 3 != 0:
            ax.set_ylabel("")

        # for columns with > 2 labels, rotate them to avoid overlap
        if len(data[column].unique()) > 2:
            ax.set_xticks(data[column].unique())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            
def plot_numeric_col(data: pd.DataFrame, columns: List[str],
                     title: str, figsize: Tuple[int, int]) -> None:
    """
    Plots histograms for specified numeric columns in a DataFrame.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    columns : List[str]
        The list of column names to be plotted.
    title : str
        The title for the entire figure.
    figsize : Tuple[int, int]
        The size of the figure (width, height).

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.99)
    for i, column in enumerate(columns):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(column.capitalize())
        sns.histplot(x=column, data=data, 
                     ax=ax, color="blue")
        ax.set_ylabel("")
        ax.set_xlabel("")
        
def hist_box_num_binary(data: pd.DataFrame, num_columns: List[str],
                        title: str, figsize: Tuple[int, int], 
                        hue: str=None) -> None:
    """
    Plots histograms and box plots for specified numeric columns, 
    differentiated by a binary categorical hue.

    Parameters:
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    num_columns : List[str]
        The list of numeric column names to be plotted.
    hue : str
        The binary categorical variable used for differentiation in the plots.
    title : str
        The title for the entire figure.
    figsize : Tuple[int, int]
        The size of the figure (width, height).

    Returns:
        None
    """
    fig, ax = plt.subplots(2, len(num_columns), figsize=figsize)
    plt.subplots_adjust(left=0, right=1,
                        top=1, bottom=0, wspace=0.2, hspace=0)
    plt.suptitle(title, fontsize=16, y=1.1)
    for i, column in enumerate(num_columns):
        if hue:
            sns.histplot(x=column, data=data, hue=hue, 
                         kde=True, ax=ax[1, i], palette=["blue", "red"])
        else:
            sns.histplot(x=column, data=data, hue=hue, 
                         kde=True, ax=ax[1, i], color="blue")
        sns.boxplot(y=hue, data=data, x=column, 
                    ax=ax[0, i], orient="h", color="blue")
        ax[0, i].set_xlabel("")
        ax[0, i].set_xticks([])
        
def make_mi_scores(X: pd.DataFrame, y: pd.Series, 
                   discrete_features: Optional[List[bool]] = None,
                   random_state: Optional[int] = None) -> pd.Series:
    """
    Calculate Mutual Information scores for features in a dataset.

    Parameters:
    ----------
    X : pd.DataFrame
        The input features as a DataFrame where each column is a feature.
    y : pd.Series
        The target variable.
    discrete_features : Optional[List[bool]], optional
        A list indicating whether each column in X is discrete (True) or continuous (False). 
        If not provided, all features are assumed to be continuous.
    random_state : Optional[int], optional
        Random seed used to ensure reproducibility of the results. 
        If not provided, results may vary each time the function is run.

    Returns:
    -------
    pd.Series
        A Series containing the MI scores for each feature, sorted in descending order.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores