import pickle


def save_object(filename, object):
    with open(filename, 'wb') as file:
        pickle.dump(object, file)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
