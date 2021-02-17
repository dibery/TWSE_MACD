import torch, sys, numpy, tqdm, pandas, multiprocessing

X_SPAN, Y_SPAN = 20, 10
FIELDS = [6, 7, 8, 9]
SPLIT = '20190101'
EPOCH = 25

def read(fname):
    price = pandas.read_csv(fname, header=None, dtype=str)
    train, test = price[price[0] < SPLIT][FIELDS], price[price[0] >= SPLIT][FIELDS]
    train = train.apply(pandas.to_numeric, errors='coerce').dropna().to_numpy(dtype=numpy.float32)
    test = test.apply(pandas.to_numeric, errors='coerce').dropna().to_numpy(dtype=numpy.float32)
    train_y = numpy.array([train[i + 1:i + Y_SPAN + 1, 1].max() for i in range(X_SPAN - 1, train.shape[0] - Y_SPAN)])
    train_x = numpy.array([train[i - X_SPAN + 1:i + 1,:].flatten() for i in range(X_SPAN - 1, train.shape[0] - Y_SPAN)])
    test_y = numpy.array([test[i + 1:i + Y_SPAN + 1, 1].max() for i in range(X_SPAN - 1, test.shape[0] - Y_SPAN)])
    test_x = numpy.array([test[i - X_SPAN + 1:i + 1,:].flatten() for i in range(X_SPAN - 1, test.shape[0] - Y_SPAN)])
    #'''
    if train_x.shape[0]:
        mean = train_x[:,-1:]
        train_x, train_y = train_x / mean - 1, train_y / mean.flatten() - 1
    if test_x.shape[0]:
        mean = test_x[:,-1:]
        test_x, test_y = test_x / mean - 1, test_y / mean.flatten() - 1
    #'''
    return (train_x.reshape((-1, len(FIELDS) * X_SPAN)).astype(numpy.float32),
            train_y.reshape((-1, 1)).astype(numpy.float32),
            test_x.reshape((-1, len(FIELDS) * X_SPAN)).astype(numpy.float32),
            test_y.reshape((-1, 1)).astype(numpy.float32))

with multiprocessing.Pool() as pool:
    train_x, train_y, test_x, test_y = pool.map(numpy.concatenate, zip(*map(read, tqdm.tqdm(sys.argv[1:], ascii=True, ncols=80, leave=False))))
    '''
    for i, j in zip(x, y):
        print(*i.flatten(), *j.flatten())
    exit()
    '''
data = torch.utils.data.DataLoader(list(zip(train_x, train_y)), 256, shuffle=True, num_workers=4)
tdata = torch.utils.data.DataLoader(list(zip(test_x, test_y)), 256, shuffle=True, num_workers=4)

class learn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(len(FIELDS) * X_SPAN, 1)
        self.l1 = torch.nn.Linear(len(FIELDS) * X_SPAN, 500)
        self.l2 = torch.nn.Linear(500, 50)
        self.l3 = torch.nn.Linear(50, 1)
        self.selu = torch.nn.SELU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, x):
        return self.l(x)
        #return self.l3(self.l2(self.l1(x)))
        x = self.l1(x)
        x = self.logsigmoid(x)
        x = self.l2(x)
        x = self.selu(x)
        x = self.l3(x)
        #x = self.tanh(x)
        return x
        return self.tanh(self.l3(self.relu(self.l2(self.sigmoid(self.l1(x))))))

model = learn().cuda()
loss = torch.nn.L1Loss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=4e-5)

for epoch in range(EPOCH):
    epoch_loss = count = error = 0
    for x, y in tqdm.tqdm(data, ncols=80, ascii=True, leave=False, desc=f'Epoch {epoch}'):
        x, y = x.cuda(), y.cuda()
        out = model(x)
        this_loss = loss(out, y)
        opt.zero_grad()
        this_loss.backward()
        opt.step()

        epoch_loss += this_loss.item()
        count += x.shape[0]
        error += (out - y).abs().sum()
        '''
        if epoch == EPOCH - 1:
            x, y, out = x.cpu(), y.cpu(), out.cpu()
            for i, j, k in zip(x, y, out):
                print(i.tolist()[-1], *j.flatten().tolist(), *k.flatten().tolist())
        '''
    print(epoch_loss / count, error.item() / count, end=' ')
    error = count = 0
    for x, y in tqdm.tqdm(tdata, ncols=80, ascii=True, leave=False, desc=f'Epoch {epoch}'):
        x, y = x.cuda(), y.cuda().flatten()
        out = model(x).flatten()
        error += (out - y).abs().sum()
        count += x.shape[0]
    print(error.item() / count)
