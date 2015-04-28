--Train Program
require 'torch'

if not opt then
  print ('==> Processing Options')
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Perceptron user input options for testing')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-theta', 0.2, 'Theta for activation function defualt is 0.2')
  cmd:option('-test', 0, 'Number of trainset to use as testset')
  cmd:option('-dataset', 'test' , 'Number of trainset to use as testset')
  cmd:text()
  opt = cmd:parse(arg or {})
end

if opt.dataset == 'test' then
  loaded = torch.load('TestData.dat')
else
  loaded = torch.load('TrainData.dat')
end

testData = {
    data = loaded.X ,
    label = loaded.T ,
    size = function() return (#testData.data)[1] end
}

theta = opt.theta
m = testData.size()     --Number of train data
n = testData.data:size(2)    --Number of features
k = testData.label:size(2)   --Number of Outputs

w = torch.Tensor(n, k):fill(0)    --matrix of weights
w = torch.load('weight.dat')
o = torch.Tensor(m,k):fill(0)



o = testData.data * w

for i = 1, m do
    for j = 1, k do
        if o[i][j] > theta then
            o[i][j] = 1
        elseif o[i][j] <= -theta then
            o[i][j] = -1
        else
            o[i][j] = 0
        end
    end
end
num = opt.test
if num == 0 then
  print(o)
else
  print('Feeding trainset number ' .. num .. ' as testset')
  print(o[num])
end

E = testData.label - o
E:pow(2)
e = torch.sum(E)
e = e % 100
print('The Error Rate is : ' .. e .. '%')