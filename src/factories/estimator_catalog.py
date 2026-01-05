"""
Estimator Catalog

This is your single source of truth that maps your YAML spec "type" strings
to the real Python class path.

Example YAML:
  estimator:
    type: "tree_clf"
    params:
      max_depth: 3

The factory will:
  - look up "tree_clf" in ESTIMATOR_CATALOG
  - import sklearn.tree.DecisionTreeClassifier dynamically
  - instantiate it with params

Design notes
- Keep keys stable (they become part of your config API).
- Prefer explicit *_clf vs *_reg when ambiguous.
- You can include aliases for backward compatibility.
"""

ESTIMATOR_CATALOG = {
    # ------------------------------------------------------------
    # Dummies
    # ------------------------------------------------------------
    "dummy_clf": "sklearn.dummy.DummyClassifier",
    "dummy_reg": "sklearn.dummy.DummyRegressor",

    # ------------------------------------------------------------
    # Linear models — Classification
    # ------------------------------------------------------------
    "logreg": "sklearn.linear_model.LogisticRegression",
    "logreg_cv": "sklearn.linear_model.LogisticRegressionCV",
    "ridge_clf": "sklearn.linear_model.RidgeClassifier",
    "ridge_clf_cv": "sklearn.linear_model.RidgeClassifierCV",
    "sgd_clf": "sklearn.linear_model.SGDClassifier",
    "perceptron": "sklearn.linear_model.Perceptron",
    "pa_clf": "sklearn.linear_model.PassiveAggressiveClassifier",

    # Linear SVM
    "linearsvc": "sklearn.svm.LinearSVC",

    # ------------------------------------------------------------
    # Linear models — Regression
    # ------------------------------------------------------------
    "linear": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "ridge_cv": "sklearn.linear_model.RidgeCV",
    "lasso": "sklearn.linear_model.Lasso",
    "lasso_cv": "sklearn.linear_model.LassoCV",
    "elasticnet": "sklearn.linear_model.ElasticNet",
    "elasticnet_cv": "sklearn.linear_model.ElasticNetCV",

    "lars": "sklearn.linear_model.Lars",
    "lars_cv": "sklearn.linear_model.LarsCV",
    "lassolars": "sklearn.linear_model.LassoLars",
    "lassolars_cv": "sklearn.linear_model.LassoLarsCV",
    "lassolars_ic": "sklearn.linear_model.LassoLarsIC",

    "omp": "sklearn.linear_model.OrthogonalMatchingPursuit",
    "omp_cv": "sklearn.linear_model.OrthogonalMatchingPursuitCV",

    "sgd": "sklearn.linear_model.SGDRegressor",

    # Robust regression
    "huber": "sklearn.linear_model.HuberRegressor",
    "theilsen": "sklearn.linear_model.TheilSenRegressor",
    "ransac": "sklearn.linear_model.RANSACRegressor",

    # Bayesian linear regression
    "bayes_ridge": "sklearn.linear_model.BayesianRidge",
    "ard": "sklearn.linear_model.ARDRegression",

    # ------------------------------------------------------------
    # Trees / Ensembles — Classification
    # ------------------------------------------------------------
    "tree_clf": "sklearn.tree.DecisionTreeClassifier",
    "rf_clf": "sklearn.ensemble.RandomForestClassifier",
    "extratrees_clf": "sklearn.ensemble.ExtraTreesClassifier",
    "ada_clf": "sklearn.ensemble.AdaBoostClassifier",
    "gboost_clf": "sklearn.ensemble.GradientBoostingClassifier",
    "hgb_clf": "sklearn.ensemble.HistGradientBoostingClassifier",
    "bagging_clf": "sklearn.ensemble.BaggingClassifier",

    # ------------------------------------------------------------
    # Trees / Ensembles — Regression
    # ------------------------------------------------------------
    "tree": "sklearn.tree.DecisionTreeRegressor",
    "rf": "sklearn.ensemble.RandomForestRegressor",
    "extratrees": "sklearn.ensemble.ExtraTreesRegressor",
    "ada": "sklearn.ensemble.AdaBoostRegressor",
    "gboost": "sklearn.ensemble.GradientBoostingRegressor",
    "hgb": "sklearn.ensemble.HistGradientBoostingRegressor",
    "bagging_reg": "sklearn.ensemble.BaggingRegressor",

    # ------------------------------------------------------------
    # SVM / Kernel — Classification
    # ------------------------------------------------------------
    "svc": "sklearn.svm.SVC",
    "nusvc": "sklearn.svm.NuSVC",

    # ------------------------------------------------------------
    # SVM / Kernel — Regression
    # ------------------------------------------------------------
    "svr": "sklearn.svm.SVR",
    "nusvr": "sklearn.svm.NuSVR",
    "linearsvr": "sklearn.svm.LinearSVR",

    # ------------------------------------------------------------
    # Neighbors — Classification
    # ------------------------------------------------------------
    "knn_clf": "sklearn.neighbors.KNeighborsClassifier",
    "radiusnn_clf": "sklearn.neighbors.RadiusNeighborsClassifier",
    "nearest_centroid": "sklearn.neighbors.NearestCentroid",

    # ------------------------------------------------------------
    # Neighbors — Regression
    # ------------------------------------------------------------
    "knn": "sklearn.neighbors.KNeighborsRegressor",
    "radiusnn_reg": "sklearn.neighbors.RadiusNeighborsRegressor",

    # ------------------------------------------------------------
    # Naive Bayes — Classification
    # ------------------------------------------------------------
    "gnb": "sklearn.naive_bayes.GaussianNB",
    "bnb": "sklearn.naive_bayes.BernoulliNB",
    "mnb": "sklearn.naive_bayes.MultinomialNB",
    "cnb": "sklearn.naive_bayes.ComplementNB",
    "categorical_nb": "sklearn.naive_bayes.CategoricalNB",

    # ------------------------------------------------------------
    # Discriminant Analysis — Classification
    # ------------------------------------------------------------
    "lda": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "qda": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",

    # ------------------------------------------------------------
    # Neural Networks
    # ------------------------------------------------------------
    "mlp_clf": "sklearn.neural_network.MLPClassifier",
    "mlp": "sklearn.neural_network.MLPRegressor",

    # ------------------------------------------------------------
    # Gaussian Process
    # ------------------------------------------------------------
    "gp_clf": "sklearn.gaussian_process.GaussianProcessClassifier",
    "gp": "sklearn.gaussian_process.GaussianProcessRegressor",

    # ------------------------------------------------------------
    # GLM-like (Tweedie family)
    # ------------------------------------------------------------
    "poisson": "sklearn.linear_model.PoissonRegressor",
    "gamma": "sklearn.linear_model.GammaRegressor",
    "tweedie": "sklearn.linear_model.TweedieRegressor",

    # ------------------------------------------------------------
    # Special regression
    # ------------------------------------------------------------
    "isotonic": "sklearn.isotonic.IsotonicRegression",
    "pls": "sklearn.cross_decomposition.PLSRegression",
    "plssvd": "sklearn.cross_decomposition.PLSSVD",

    # ------------------------------------------------------------
    # Meta-estimators / wrappers (useful later)
    # ------------------------------------------------------------
    "calibrated_clf": "sklearn.calibration.CalibratedClassifierCV",
    "ovr": "sklearn.multiclass.OneVsRestClassifier",
    "ovo": "sklearn.multiclass.OneVsOneClassifier",
    "ecoc": "sklearn.multiclass.OutputCodeClassifier",

    "multioutput_clf": "sklearn.multioutput.MultiOutputClassifier",
    "multioutput_reg": "sklearn.multioutput.MultiOutputRegressor",
    "classifier_chain": "sklearn.multioutput.ClassifierChain",
    "regressor_chain": "sklearn.multioutput.RegressorChain",

    "voting_clf": "sklearn.ensemble.VotingClassifier",
    "voting_reg": "sklearn.ensemble.VotingRegressor",
    "stacking_clf": "sklearn.ensemble.StackingClassifier",
    "stacking_reg": "sklearn.ensemble.StackingRegressor",
}
