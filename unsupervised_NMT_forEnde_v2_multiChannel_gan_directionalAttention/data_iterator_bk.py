import cPickle as pkl
import gzip
import numpy


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class domainTextIterator:
    def __init__(self, s_domain_data, t_domain_data, g_domain_data, dic, batch=1, maxlen=50, n_words_target=-1):
         self.s_domain_data = fopen(s_domain_data, 'r')
         self.t_domain_data = fopen(t_domain_data, 'r')
         self.g_domain_data = fopen(g_domain_data, 'r')

         with open(dic) as f_trg:
            self.dic_target = pkl.load(f_trg)

         self.batch_size = batch
         assert self.batch_size % 2 == 0
         self.maxlen = maxlen
         self.n_words_trg = n_words_target
         self.end_of_data = False

    def __iter__(self):
         return self

    def reset(self):
         self.s_domain_data.seek(0)
         self.t_domain_data.seek(0)
         self.g_domain_data.seek(0)
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        x = []
        y = []

        try:
            while True:
                ss = self.s_domain_data.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dic_target[w] if w in self.dic_target else 1 for w in ss]
                if self.n_words_trg > 0:
                    ss = [w if w < self.n_words_trg else 1 for w in ss]

                tt = self.t_domain_data.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dic_target[w] if w in self.dic_target else 1 for w in tt]
                if self.n_words_trg > 0:
                    tt = [w if w < self.n_words_trg else 1 for w in tt]

                gg = self.g_domain_data.readline()
                if gg == "":
                    raise IOError
                gg = gg.strip().split()
                gg = [self.dic_target[w] if w in self.dic_target else 1 for w in gg]
                if self.n_words_trg > 0:
                    gg = [w if w < self.n_words_trg else 1 for w in gg]

                if len(ss) > self.maxlen or len(tt) >self.maxlen or len(gg) > self.maxlen:
                    continue

                x.append(ss)
                y.append([1,0,0])
                x.append(tt)
                y.append([0,1,0])
                x.append(gg)
                y.append([0,0,1])

                if len(x) >= self.batch_size and len(y) >= self.batch_size:
                    shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                    x_np = numpy.array(x)
                    y_np = numpy.array(y)

                    x_np_shuffled = x_np[shuffle_indices]
                    y_np_shuffled = y_np[shuffle_indices]

                    x_shuffled = x_np_shuffled.tolist()
                    y_shuffled = y_np_shuffled.tolist()

                    break
        except IOError:
            self.end_of_data = True

        if len(x) <=0 or len(y) <=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        if len(x) >=self.batch_size:     
            return x_shuffled[:self.batch_size], y_shuffled[:self.batch_size]
        else:
            return x, y

class pairTextIterator:
    def __init__(self, source_data, pair_data, npair_data, generated_data, dic, dic_source, batch=1, maxlen=50, n_words_target=-1, n_words_src=-1):
         self.source_data = fopen(source_data, 'r')
         self.pair_data = fopen(pair_data, 'r')
         self.npair_data = fopen(npair_data, 'r')
         self.generated_data = fopen(generated_data, 'r')

         with open(dic) as f_trg:
            self.dic_target = pkl.load(f_trg)

         with open(dic_source) as s_trg:
            self.dic_source = pkl.load(s_trg)

         self.batch_size = batch
         assert self.batch_size % 2 == 0

         self.maxlen = maxlen
         self.n_words_trg = n_words_target
         self.n_words_src = n_words_src
         self.end_of_data = False

    def __iter__(self):
         return self

    def reset(self):
         self.source_data.seek(0)
         self.pair_data.seek(0)
         self.npair_data.seek(0)
         self.generated_data.seek(0)
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        s = []
        x = []
        y = []

        try:
            while True:
                source = self.source_data.readline()
                if source == "":
                    raise IOError
                source = source.strip().split()
                source = [self.dic_source[w] if w in self.dic_source else 1 for w in source]
                if self.n_words_src > 0:
                    source = [w if w < self.n_words_src else 1 for w in source]

                ss = self.pair_data.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dic_target[w] if w in self.dic_target else 1 for w in ss]
                if self.n_words_trg > 0:
                    ss = [w if w < self.n_words_trg else 1 for w in ss]

                tt = self.npair_data.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dic_target[w] if w in self.dic_target else 1 for w in tt]
                if self.n_words_trg > 0:
                    tt = [w if w < self.n_words_trg else 1 for w in tt]

                gg = self.generated_data.readline()
                if gg == "":
                    raise IOError
                gg = gg.strip().split()
                gg = [self.dic_target[w] if w in self.dic_target else 1 for w in gg]
                if self.n_words_trg > 0:
                    gg = [w if w < self.n_words_trg else 1 for w in gg]

                if len(source) > self.maxlen or len(ss) > self.maxlen or len(tt) >self.maxlen or len(gg) > self.maxlen:
                    continue

                s.append(source)
                x.append(ss)
                y.append([1,0,0])

                s.append(source)
                x.append(tt)
                y.append([0,1,0])

                s.append(source)
                x.append(gg)
                y.append([0,0,1])

                if len(s) >=self.batch_size and len(x) >= self.batch_size and len(y) >= self.batch_size:
                    shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                    s_np = numpy.array(s)
                    x_np = numpy.array(x)
                    y_np = numpy.array(y)

                    s_np_shuffled = s_np[shuffle_indices]
                    x_np_shuffled = x_np[shuffle_indices]
                    y_np_shuffled = y_np[shuffle_indices]

                    s_shuffled = s_np_shuffled.tolist()
                    x_shuffled = x_np[shuffle_indices]
                    y_shuffled = y_np_shuffled.tolist()

                    break
        except IOError:
            self.end_of_data = True

        if len(s) <=0 or len(x) <=0 or len(y) <=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        if len(s) >= self.batch_size:    
            return s_shuffled[:self.batch_size], x_shuffled[:self.batch_size], y_shuffled[:self.batch_size]
        else:
            return s, x, y

class disThreeTextIterator:
    def __init__(self, positive_data, negative_data, source_data, dic_target, dic_source, batch=1, maxlen=50, n_words_target=-1, n_words_source=-1):
         self.positive = fopen(positive_data, 'r')
         self.negative = fopen(negative_data, 'r')
         self.source = fopen(source_data, 'r')

         with open(dic_target) as f_trg:
            self.dic_target = pkl.load(f_trg)
         with open(dic_source) as s_trg:
            self.dic_source = pkl.load(s_trg)

         self.batch_size = batch
         assert self.batch_size % 2 == 0
         self.maxlen = maxlen
         self.n_words_trg = n_words_target
         self.n_words_src = n_words_source
         self.end_of_data = False

    def __iter__(self):
         return self

    def reset(self):
         self.positive.seek(0)
         self.negative.seek(0)
         self.source.seek(0)
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        positive = []
        negative = []
        source = []
        x = []
        xs = []
        y = []

        try:
            while True:
                ss = self.positive.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dic_target[w] if w in self.dic_target else 1 for w in ss]
                if self.n_words_trg > 0:
                    ss = [w if w < self.n_words_trg else 1 for w in ss]

                tt = self.negative.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dic_target[w] if w in self.dic_target else 1 for w in tt]
                if self.n_words_trg > 0:
                    tt = [w if w < self.n_words_trg else 1 for w in tt]

                ll = self.source.readline()
                if ll == "":
                    raise IOError
                ll = ll.strip().split()
                ll = [self.dic_source[w] if w in self.dic_source else 1 for w in ll]
                if self.n_words_src > 0:
                    ll = [w if w < self.n_words_src else 1 for w in ll]

                if len(ss) > self.maxlen or len(tt) >self.maxlen or len(ll) > self.maxlen:
                    continue

                positive.append(ss)
                negative.append(tt)
                source.append(ll)

                x = positive + negative

                positive_labels = [[0, 1] for _ in positive]
                negative_labels = [[1, 0] for _ in negative]
                y = positive_labels + negative_labels

                xs = source + source

                shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                x_np = numpy.array(x)
                y_np = numpy.array(y)
                xs_np =numpy.array(xs)

                x_np_shuffled = x_np[shuffle_indices]
                y_np_shuffled = y_np[shuffle_indices]
                xs_np_shuffled = xs_np[shuffle_indices]

                x_shuffled = x_np_shuffled.tolist()
                y_shuffled = y_np_shuffled.tolist()
                xs_shuffled =xs_np_shuffled.tolist()

                if len(x_shuffled) >= self.batch_size and len(y_shuffled) >= self.batch_size and len(xs_shuffled) >=self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(positive) <=0 or len(negative) <=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
            
        return x_shuffled, y_shuffled, xs_shuffled

class disTextIterator:
    def __init__(self, positive_data, negative_data, dis_dict, batch=1, maxlen=30, n_words_target=-1):
         self.positive = fopen(positive_data, 'r')
         self.negative = fopen(negative_data, 'r')
         with open(dis_dict) as f:
            self.dis_dict = pkl.load(f)

         self.batch_size = batch
         assert self.batch_size % 2 == 0, 'the batch size of disTextIterator is not an even number'
        
         self.maxlen = maxlen
         self.n_words_target = n_words_target
         self.end_of_data = False

    def __iter__(self):
         return self
    
    def reset(self):
         self.positive.seek(0)
         self.negative.seek(0)

    def next(self):
     if self.end_of_data:
          self.end_of_data = False
          self.reset()
          raise StopIteration

     positive = []
     negative = []
     x = []
     y = []
     try:
        while True:
            ss = self.positive.readline()
            if ss == "":
               raise IOError
            ss = ss.strip().split()
            ss = [self.dis_dict[w] if w in self.dis_dict else 1 for w in ss]
            if self.n_words_target > 0:
               ss = [w if w < self.n_words_target else 1 for w in ss]
                    
               
            tt = self.negative.readline()
            if tt == "":
               raise IOError
            tt = tt.strip().split()
            tt = [self.dis_dict[w] if w in self.dis_dict else 1 for w in tt]
            if self.n_words_target > 0:
               tt = [w if w < self.n_words_target else 1 for w in tt]

            if len(ss) > self.maxlen or len(tt) > self.maxlen:
               continue

            positive.append(ss)
            negative.append(tt)
            x = positive + negative
            positive_labels = [[0, 1] for _ in positive]
            negative_labels = [[1, 0] for _ in negative]
            y = positive_labels + negative_labels
            shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
            x_np = numpy.array(x)
            y_np = numpy.array(y)
            x_np_shuffled = x_np[shuffle_indices]
            y_np_shuffled = y_np[shuffle_indices]

            x_shuffled = x_np_shuffled.tolist()
            y_shuffled = y_np_shuffled.tolist()

            if len(x_shuffled) >= self.batch_size and len(y_shuffled) >= self.batch_size:
                break

     except IOError:
        self.end_of_data = True 

     if len(positive) <= 0 or len(negative) <= 0:
        self.end_of_data = False
        self.reset()
        raise StopIteration
         
     return x_shuffled, y_shuffled

           
class genTextIterator:
    def __init__(self, train_data, source_dict, batch_size=1, maxlen=30, n_words_source=-1):
        self.source = fopen(train_data, 'r')

        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data= False
            self.reset()
            raise StopIteration

        source = []
        try:
            while True:
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w<self.n_words_source else 1 for w in ss]

                if len(ss) > self.maxlen:
                    continue

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except:
            self.end_of_data=True

        if len(source)<=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
