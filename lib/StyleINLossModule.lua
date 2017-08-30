require 'nn'
local module, parent = torch.class('nn.StyleINLossModule', 'nn.Module')

function module:__init(strength, normalize, nChannel)
    parent.__init(self)
    self.normalize = normalize or false
    self.strength = strength or 1
    self.target_mean = nil
    self.target_std = nil
    self.mean_loss = 0
    self.std_loss = 0
    self.loss = 0
    self.nC = nChannel

    self.std_net = nn.Sequential() -- assume the input is centered
    self.std_net:add(nn.Square())
    self.std_net:add(nn.Mean(3))
    self.std_net:add(nn.Sqrt(1e-6))
    self.mean_net = nn.Sequential()
    self.mean_net:add(nn.Mean(3))

    self.mean_criterion = nn.MSECriterion()
    self.mean_criterion.sizeAverage = false
    self.std_criterion = nn.MSECriterion()
    self.std_criterion.sizeAverage = false

    self.std_net = self.std_net:cuda()
    self.mean_net = self.mean_net:cuda()
    self.mean_criterion = self.mean_criterion:cuda()
    self.std_criterion = self.std_criterion:cuda()
end

function module:clearState()
    self.std_net:clearState()
    self.mean_net:clearState()
    return parent.clearState(self)
end

function module:setTarget(target_features)
    if target_features:nDimension() == 3 then
        local C = target_features:size(1)
        target_features = target_features:view(1, C, -1)
    elseif target_features:nDimension() == 4 then
        local N,C = target_features:size(1), target_features:size(2)
        target_features = target_features:view(N, C, -1)
    else
        error('Target must be 3D or 4D')
    end
    self.target_mean = torch.mean(target_features, 3) -- N*C*1
    self.target_std = torch.std(target_features, 3, true)
    return self
end

function module:unsetTarget()
    self.target_mean = nil
    self.target_std = nil
    return self
end

function module:updateOutput(input)
    self.output = input
    if self.target_mean ~= nil and self.target_std ~= nil then

        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
        assert(input:size(2) == self.target_mean:size(2))
        assert(input:size(2) == self.target_std:size(2))

        local input_view = input:view(N, C, -1)
        if N < self.target_mean:size(1) then
            self.target_mean = self.target_mean[1]
            self.target_std = self.target_std[1]
        elseif N > self.target_mean:size(1) then
            self.target_mean = self.target_mean:expand(N,C,1)
            self.target_std = self.target_std:expand(N,C,1)
        end

        self.input_mean = self.mean_net:forward(input_view)
        self.input_centered = torch.add(input_view, -self.input_mean:view(N, C, 1):expand(N, C, H*W)) -- centered input
        self.input_std = self.std_net:forward(self.input_centered)

        self.mean_loss = self.mean_criterion:forward(self.input_mean, self.target_mean)
        self.std_loss = self.std_criterion:forward(self.input_std, self.target_std)
        self.mean_loss = self.mean_loss / N -- normalized w.r.t. batch size
        self.std_loss = self.std_loss / N -- normalized w.r.t. batch size
        self.loss = self.mean_loss + self.std_loss
        self.loss = self.loss * self.strength
    end
    return self.output
end

function module:updateGradInput(input, gradOutput)
    if self.target_mean ~= nil and self.target_std ~= nil then
        local nInputDim = input:nDimension()
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
        assert(input:size(2) == self.target_mean:size(2))
        assert(input:size(2) == self.target_std:size(2))

        local input_view = input:view(N, C, -1)
        local mean_grad = self.mean_criterion:backward(self.input_mean, self.target_mean)
        local std_grad = self.std_criterion:backward(self.input_std, self.target_std)
        self.gradInput = self.mean_net:backward(input_view, mean_grad)
        local std_gradInput_centered = self.std_net:backward(self.input_centered, std_grad)
        local std_gradInput = std_gradInput_centered:add(-std_gradInput_centered:mean(3):expand(N, C, H*W))
        self.gradInput:add(std_gradInput)
        self.gradInput = self.gradInput:view(N,C,H,W)
        self.gradInput:div(N) -- normalize w.r.t. batch size

        if self.normalize then
            self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
        end

        if nInputDim == 3 then
            self.gradInput = self.gradInput:view(C,H,W)
        end

        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput = gradOutput
    end
    return self.gradInput
end
