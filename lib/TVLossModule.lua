require 'lib/TVLossCriterion'

local module, parent = torch.class('nn.TVLossModule', 'nn.Module')

function module:__init(strength)
    parent.__init(self)
    self.strength = strength or 1
    self.crit = nn.TVLossCriterion()
    self.loss = 0
end

function module:updateOutput(input)
    self.loss = self.crit:forward(input)
    self.loss = self.loss * self.strength
    self.output = input
    return self.output
end

function module:updateGradInput(input, gradOutput)
    self.gradInput = self.crit:backward(input)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
end
