require 'nn'

local TVLossCriterion, parent = torch.class('nn.TVLossCriterion', 'nn.Criterion')

function TVLossCriterion:__init()
    parent.__init(self)

    local crop_l = nn.SpatialZeroPadding(-1, 0, 0, 0)
    local crop_r = nn.SpatialZeroPadding(0, -1, 0, 0)
    local crop_t = nn.SpatialZeroPadding(0, 0, -1, 0)
    local crop_b = nn.SpatialZeroPadding(0, 0, 0, -1)
    self.target = torch.zeros(1)
    self.mse = nn.MSECriterion()
    self.mse.sizeAverage = false

    local lr = nn.Sequential()
    lr:add(nn.ConcatTable():add(crop_l):add(crop_r))
    lr:add(nn.CSubTable())
    local tb = nn.Sequential()
    tb:add(nn.ConcatTable():add(crop_t):add(crop_b))
    tb:add(nn.CSubTable())

    self.crit = nn.ConcatTable():add(lr):add(tb)
end

function TVLossCriterion:updateOutput(input)
    local output = self.crit:forward(input)
    local loss = 0
    for i=1,2 do
        local target
        if output[i]:nDimension() == 3 then
            target = self.target:view(1,1,1):expandAs(output[i])
        else
            target = self.target:view(1,1,1,1):expandAs(output[i])
        end
        loss = loss + self.mse:forward(output[i], target)
    end
    self.output = loss
    return self.output
end

function TVLossCriterion:updateGradInput(input)
    self.gradInput:resizeAs(input):zero()
    local output = self.crit.output
    local df_do = {}
    for i=1,2 do
        local target
        if output[i]:nDimension() == 3 then
            target = self.target:view(1,1,1):expandAs(output[i])
        else
            target = self.target:view(1,1,1,1):expandAs(output[i])
        end
        df_do[i] = self.mse:backward(output[i], target):clone()
    end
    local grad = self.crit:backward(input, df_do)
    self.gradInput:copy(self.crit:backward(input, df_do))
    return self.gradInput
end
