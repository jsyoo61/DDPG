import pickle

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def write(path, content, encoding = None):
    with open(path, 'w', encoding = encoding) as f:
        f.write(content)

def read_text(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.read()
    return text

def print_stars():
    print('*' * 50)

def multiply_tuple(tup, number):

    multiplied_tuple = list()
    for entry in tup:
        multiplied_tuple.append(entry * number)

    return tuple(multiplied_tuple)


class Printer():
    def __init__(self):
        self.content = ''

    def add(self, text, end = '\n'):
        self.content += text + end

    def print(self):
        print(self.content)

    def reset(self):
        self.content = ''
