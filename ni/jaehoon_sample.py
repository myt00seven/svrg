#######################3  Theano shared dataset ######################
def shared_dataset(data_xy, borrow=True):

   data_x, data_y = data_xy
   shared_x = theano.shared(numpy.asarray(data_x,
                                          dtype=theano.config.floatX),
                            borrow=borrow)
   shared_y = theano.shared(numpy.asarray(data_y,
                                          dtype=theano.config.floatX),
                            borrow=borrow)
   return shared_x, theano.tensor.cast(shared_y, 'int32')


start_time = timeit.default_timer()
DATA_PATH = "/Users/myt007/git/svrg/ni/"

train_x_raw = gzip.open(DATA_PATH+"/train_x.txt.gz", 'rb')
train_x = pickle.load(train_x_raw)
train_y_raw = gzip.open(DATA_PATH+"/train_y.txt.gz", 'rb')
train_y = pickle.load(train_y_raw)
test_x_raw = gzip.open(DATA_PATH+"/test_x.txt.gz", 'rb')
test_x = pickle.load(test_x_raw)
test_y_raw = gzip.open(DATA_PATH+"/test_y.txt.gz", 'rb')
test_y = pickle.load(test_y_raw)

train_set = (train_x, train_y)
valid_set = (train_x, train_y)
test_set = (test_x, test_y)

train_set_x, train_set_y = shared_dataset(train_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
test_set_x, test_set_y = shared_dataset(test_set)

############################################################################
datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
numpy_rng = numpy.random.RandomState(123)
print '... building the model'
# construct the Deep Belief Network
dbn = DBN(numpy_rng=numpy_rng, n_ins=41,
         hidden_layers_sizes=[15],
         n_outs=5) (edited)