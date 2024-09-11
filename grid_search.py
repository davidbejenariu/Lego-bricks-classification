from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


class ClusteringGridSearch(BaseEstimator, ClusterMixin):
    def __init__(self, estimator, param_grid, scoring=silhouette_score):
        self.best_params_ = None
        self.best_estimator_ = None
        self.scores_ = None
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring

    def fit(self, X, y):
        params_combinations = ParameterGrid(self.param_grid)

        # Dictionary to store scores for each hyperparameter combination
        scores_dict = {}

        for i, params in enumerate(params_combinations):
            model = self.estimator.set_params(**params)
            model.fit(X)

            # Evaluate the model using the specified scoring metric
            score = self.scoring(X, model.labels_)
            scores_dict[str(params)] = score

            # Print the outcome
            davies_bouldin = davies_bouldin_score(X, model.labels_)
            arand = adjusted_rand_score(y, model.labels_)
            print(f'Combination {i + 1}/{len(params_combinations)} - Silhouette: {score}, Davies-Bouldin: {davies_bouldin}, Adjusted Rand: {arand} - Params: {params}')

        # Find the best hyperparameters based on the highest score
        best_params = max(scores_dict, key=scores_dict.get)
        self.best_estimator_ = self.estimator.set_params(**eval(best_params))
        self.best_params_ = eval(best_params)
        self.scores_ = scores_dict

        return self
