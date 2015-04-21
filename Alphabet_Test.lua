--Train Program
require 'torch'

loaded = torch.load('TrainData.dat')
trainData = {
    data = loaded.X ,
    label = loaded.T ,
    size = function() return (#trainData.data)[1] end
}

if not opt then
  print ('==> Processing Options')
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Perceptron user input options for testing')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-theta', 0.2, 'Theta for activation function defualt is 0.2')
  cmd:option('-test', 1, 'Number of trainset to use as testset')
  cmd:text()
  opt = cmd:parse(arg or {})
end

theta = opt.theta
m = trainData.size()     --Number of train data
n = trainData.data:size(2)    --Number of features
k = trainData.label:size(2)   --Number of Outputs

w = torch.Tensor(n, k):fill(0)    --matrix of weights
w = torch.load('weight.dat')
o = torch.Tensor(m,k):fill(0)



o = trainData.data * w

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
--y = torch.Tensor(label)
--E = y - o
--print(E)