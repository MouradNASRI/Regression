from sklearn.svm import SVR, LinearSVR

KERNEL_REGISTRY = {
    "svr": SVR,
    "linearsvr": LinearSVR,
}
