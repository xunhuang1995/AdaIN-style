require 'lib/InstanceNormalization.lua'

local module, parent = torch.class('nn.ContentLossModule', 'nn.Module')

function module:__init(strength, normalize, nChannel)
    parent.__init(self)
    self.normalize = normalize or false
    self.strength = strength or 1
    self.target = nil
    self.loss = 0
    self.nC = nChannel
    self.crit = nn.MSECriterion()
end

function module:setTarget(target_features)
    if target_features:nDimension()==3 then
        local C,H,W = target_features:size(1), target_features:size(2), target_features:size(3)
        target_features = target_features:view(1,C,H,W)
    end
    self.target = target_features:clone()
end

function module:unsetTarget()
    self.target = nil
end

function module:updateOutput(input)
    self.output = input
    if self.target ~= nil then
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)
        local N,C,H,W = self.target:size(1), self.target:size(2), self.target:size(3), self.target:size(4)
        assert(input:isSameSizeAs(self.target),
            string.format('Input size (%d x %d x %d x %d) does not match target size (%d x %d x %d x %d)',
                input:size(1),input:size(2),input:size(3),input:size(4),
                N,C,H,W))
		self.loss = self.crit:forward(input, self.target)
        self.loss = self.loss * self.strength
    end
    return self.output
end

function module:updateGradInput(input, gradOutput)
    if self.target ~= nil then
        local nInputDim = input:nDimension()
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)
        local N,C,H,W = self.target:size(1), self.target:size(2), self.target:size(3), self.target:size(4)
        assert(input:isSameSizeAs(self.target),
            string.format('Input size (%d x %d x %d x %d) does not match target size (%d x %d x %d x %d)',
                input:size(1),input:size(2),input:size(3),input:size(4),
                N,C,H,W))
        self.gradInput = self.crit:backward(input, self.target):clone()

        if self.normalize then
            self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
        end

        if nInputDim == 3 then
            local C,H,W = input:size(2), input:size(3), input:size(4)
            self.gradInput = self.gradInput:view(C,H,W)
        end

        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput = gradOutput
    end
    return self.gradInput
end
