# Just

from collections import defaultdict

class customdict(defaultdict):

    def __add__(self, other):
        assert(issubclass(type(other),dict))

        new = defaultdict(self)
        for key, val in other.iteritems():
            new[key]+=val

    def update(self,other):
        assert(issubclass(type(other),dict))

        for key, val in other.iteritems():
            self[key]+= val