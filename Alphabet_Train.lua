--Train Program
require 'torch'

if not opt then
  print ('==> Processing Options')
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Perceptron user input options')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-alpha', 1, 'Learning rate alpha( 0 < alpha <= 1 )')
  cmd:option('-theta', 0.2, 'Theta for activation function defualt is 0.2')
  cmd:option('-save', true, 'Save Weights in weight.dat')
  cmd:text()
  opt = cmd:parse(arg or {})
end

alpha = tonumber(opt.alpha)
print('Learning Rate is set to : ' .. alpha)
theta = tonumber(opt.theta)
print('Activation Theta is set to : ' .. theta)
epoch = 0
--loading training data
loaded = torch.load('data.dat')
trainData = {
    data = loaded.x ,
    label = loaded.y ,
    size = function() return (#trainData.data) end
}

m = trainData.size()     --Number of train data
n = trainData.data[1]:size(1)    --Number of features
k = trainData.label[1]:size(1)   --Number of Outputs

w = torch.Tensor(n, k):fill(0)    --matrix of weights
out = torch.Tensor(m,k):fill(0)   --output matrix
delta_w = torch.Tensor(n, k):fill(0)    --difference of weights

flaq = true

function normal() do
    for i = 1, m do
        for j = 1, k do
            for l = 1, n do
                out[i][j] = out[i][j] + (trainData.data[i][l] * w[l][j])
            end
        end
    end
    for i = 1, m do
        for j = 1, k do
            if out[i][j] > theta then
                out[i][j] = 1
            elseif out[i][j] <= -theta then
                out[i][j] = -1
            else
                out[i][j] = 0
            end
        end
    end
end
end

temp = 0

while flaq do
    for f = 1, m do    --loop for Number of train data
        normal()
        for i = 1, n do    --loop for Number of Features
                for j = 1, k do   --lopp for Number of outputs
                    if out[f][j] ~= trainData.label[f][j] then
                        temp = w[i][j] + (alpha * (trainData.data[f][i] * trainData.label[f][j]))
                        delta_w[i][j] = temp - w[i][j]
                        w[i][j] = temp
                    else
                      delta_w[i][j] = 0
                    end
                end
        end
    end
    s = torch.sum(delta_w)
    if s == 0 then
        flaq = false
    end
    epoch = epoch + 1
    print('epoch : ' .. epoch)
end
if opt.save then
  torch.save('weight.dat', w)
end
print(w)