--Train Program
require 'torch'

loaded = torch.load('data.dat')
trainData = {
    data = loaded.x ,
    label = loaded.y ,
    size = function() return (#trainData.data) end
}

if not opt then
  print ('==> Processing Options')
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Perceptron user input options for testing')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-theta', 0.2, 'Theta for activation function defualt is 0.2')
  cmd:option('-train', 1, 'Number of trainset to use as testset')
  cmd:text()
  opt = cmd:parse(arg or {})
end

theta = opt.theta
m = trainData.size()     --Number of train data
n = trainData.data[1]:size(1)    --Number of features
k = trainData.label[1]:size(1)   --Number of Outputs

w = torch.Tensor(n, k):fill(0)    --matrix of weights
w = torch.load('weight.dat')
o = torch.Tensor(m,k):fill(0)
for i = 1, m do
    for j = 1, k do
        for l = 1, n do
            o[i][j] = o[i][j] + (trainData.data[i][l] * w[l][j])
        end
    end
end
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
num = opt.train
print('Feeding trainset number ' .. num .. ' as testset')
print(o[num])