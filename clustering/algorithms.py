from sklearn.cluster import KMeans as KMeansSK, DBSCAN as DBSCANSK



class KMeans:
    name = "KMeans"    

    def __init__(self):
        pass
    @staticmethod
    def make_instance(**kwargs):
        return KMeansSK(**kwargs)

class DBSCAN:
    name = "DBSCAN"

    def __init__(self):
        pass
    @staticmethod
    def make_instance(**kwargs):
        return DBSCANSK(**kwargs)


def algorithms():
    return {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN
    }
