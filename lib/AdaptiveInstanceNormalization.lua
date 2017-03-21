require 'nn'

--[[
Implements adaptive instance normalization (AdaIN) as described in the paper:

Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
Xun Huang, Serge Belongie
]]

local AdaptiveInstanceNormalization, parent = torch.class('nn.AdaptiveInstanceNormalization', 'nn.Module')

function AdaptiveInstanceNormalization:__init(nOutput, disabled, eps)
    parent.__init(self)

    self.eps = eps or 1e-5

    self.nOutput = nOutput
    self.batchSize = -1
    self.disabled = disabled
end

function AdaptiveInstanceNormalization:updateOutput(input) --{content, style}
    local content = input[1]
    local style = input[2]

    if self.disabled then
        self.output = content
        return self.output
    end        

    local N, Hc, Wc, Hs, Ws
    if content:nDimension() == 3 then
        assert(content:size(1) == self.nOutput)
        assert(style:size(1) == self.nOutput)
        N = 1
        Hc, Wc = content:size(2), content:size(3)
        Hs, Ws = style:size(2), style:size(3)
        content = content:view(1, self.nOutput, Hc, Wc)
        style = style:view(1, self.nOutput, Hs, Ws)
    elseif content:nDimension() == 4 then
        assert(content:size(1) == style:size(1))
        assert(content:size(2) == self.nOutput)
        assert(style:size(2) == self.nOutput)
        N = content:size(1)
        Hc, Wc = content:size(3), content:size(4)
        Hs, Ws = style:size(3), style:size(4)
    end

    -- compute target mean and standard deviation from the style input
    local styleView = style:view(N, self.nOutput, Hs*Ws)
    local targetStd = styleView:std(3, true):view(-1)
    local targetMean = styleView:mean(3):view(-1)

    -- construct the internal BN layer
    if N ~= self.batchSize or (self.bn and self:type() ~= self.bn:type()) then
        self.bn = nn.SpatialBatchNormalization(N * self.nOutput, self.eps)
        self.bn:type(self:type())
        self.batchSize = N
    end

    -- set affine params for the internal BN layer
    self.bn.weight:copy(targetStd)
    self.bn.bias:copy(targetMean)

    local contentView = content:view(1, N * self.nOutput, Hc, Wc)
    self.bn:training()
    self.output = self.bn:forward(contentView):viewAs(content)
    return self.output
end

function AdaptiveInstanceNormalization:updateGradInput(input, gradOutput)
    -- Not implemented
    self.gradInput = nil
    return self.gradInput
end

function AdaptiveInstanceNormalization:clearState()
    self.output = self.output.new()
    self.gradInput[1] = self.gradInput[1].new()
    self.gradInput[2] = self.gradInput[2].new()
    if self.bn then self.bn:clearState() end
end
