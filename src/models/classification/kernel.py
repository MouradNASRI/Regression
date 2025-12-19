"""
Classification - Kernel-based models registry.
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge  # not a classifier; do NOT include

KERNEL_REGISTRY = {
    "svc": SVC,
    "linearsvc": LinearSVC,
}
