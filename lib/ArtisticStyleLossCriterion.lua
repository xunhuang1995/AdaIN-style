require 'nn'
require 'lib/ContentLossModule'
require 'lib/StyleINLossModule'
require 'lib/TVLossModule'
require 'lib/utils'

local criterion, parent = torch.class('nn.ArtisticStyleLossCriterion', 'nn.Criterion')

function criterion:__init(cnn, layers, weights, normalize)
    parent.__init(self)

    layers = layers or {}
    layers.content = layers.content or {}
    layers.style = layers.style or {}

    weights = weights or {}
    weights.content = weights.content or 0
    weights.style = weights.style or 0
    weights.tv = weights.tv or 0

    if weights.style <= 0 then
        layers.style = {}
    end
    if weights.content <= 0 then
        layers.content = {}
    end

    assert(#layers.content ==1,
        'Should have only one content layer')

    local net = nn.Sequential()
    local style_layers = {}
    local content_layers = {}
    local next_style_idx = 1
    local next_content_idx = 1

    -- Build encoder
    if weights.tv > 0 then
        local tv_mod = nn.TVLossModule(weights.tv)
        net:add(tv_mod)
    end
    local nop = function() end
    local prevC
    for i=1,cnn:size() do
        if next_style_idx <= #layers.style or
            next_content_idx <= #layers.content then -- STOP if all loss modules have been inserted
            local layer = cnn:get(i)
            local name = layer.name
            if torch.type(layer) == 'nn.SpatialConvolution' then
                -- Remove weight gradients because the encoder weights should be fixed.
                layer.accGradParameters = nop
                layer.gradWeight = nil
                layer.gradBias = nil
                prevC = layer.nOutputPlane
            end
            net:add(layer)

            -- Add loss modules
            if layers.style[next_style_idx] ~= nil and name == layers.style[next_style_idx] then
                local loss_module = nn.StyleINLossModule(weights.style, normalize, prevC)
                net:add(loss_module)
                table.insert(style_layers, loss_module)
                next_style_idx = next_style_idx + 1
            end
            if layers.content[next_content_idx] ~= nil and name == layers.content[next_content_idx] then
                local loss_module = nn.ContentLossModule(weights.content, normalize, prevC)
                net:add(loss_module)
                table.insert(content_layers, loss_module)
                next_content_idx = next_content_idx + 1
            end
        end
    end

    --  Error checking
    if next_style_idx < #layers.style then
        error('Could not find layer ' .. layers.style[next_style_idx])
    end
    if next_content_idx < #layers.content then
        error('Could not find layer ' .. layers.content[next_content_idx])
    end

    --  Prepare
    self.net = net
    self.style_layers = style_layers
    self.content_layers = content_layers

    self.dy = torch.Tensor()
end

function criterion:setTargets(targets)
    if targets.style == nil and targets.content == nil then
        error('Must provide either target.style or target.content images.')
    end
    self:unsetTargets()

    if targets.style ~= nil then
        self:setStyleTarget(targets.style)
    end
    if targets.content ~= nil then
        self:setContentTarget(targets.content)
    end
end

function criterion:setContentTarget(target)
    if #self.content_layers == 0 then return end
    if target == nil then
        error('Must provide target content image.')
    end
    assert(target:nDimension()==3 or target:nDimension()==4, 'Content target must be 3D or 4D (batch).')
    self.targets = self.targets or {}
    self.targets.content = target:clone()
    self.net:clearState()
    self.net:forward(self.targets.content)
    for i=1,#self.content_layers do
        local target_features = self.content_layers[i].output
        self.content_layers[i]:setTarget(target_features)
    end
end

function criterion:setStyleTarget(target)
    if #self.style_layers <= 0 then return end
    if target == nil then
        error('Must provide target style image.')
    end
    assert(target:nDimension()==3 or target:nDimension()==4, 'Content target must be 3D or 4D (batch).')
    self.targets = self.targets or {}
    self.targets.style = target:clone()

    -- temporarily remove content targets, else the module
    -- may error out due to incorrect size.
    local content_targets = {}
    for i=1,#self.content_layers do
        content_targets[i] = self.content_layers[i].target
        self.content_layers[i].target = nil
    end

    self.net:clearState()
    self.net:forward(self.targets.style)
    for i=1,#self.style_layers do
        local target_features = self.style_layers[i].output
        self.style_layers[i]:setTarget(target_features)
    end

    -- reset the content targets
    for i=1,#self.content_layers do
        self.content_layers[i].target = content_targets[i]
    end
end

function criterion:unsetTargets()
    for i=1,#self.style_layers do
        self.style_layers[i]:unsetTarget()
    end
    for i=1,#self.content_layers do
        self.content_layers[i]:unsetTarget()
    end
end

--[[
Assumes input and target are both C x H x W images. (C=3)
Batch mode optional.
--]]
function criterion:updateOutput(input, targets)
    self.recompute_gradInput = true
    if not self.targets then self:setTargets(targets) end
    self.net:forward(input)
    -- accumulate losses from the style loss layers
    local styleLoss = 0
    local contentLoss = 0
    for _, mod in ipairs(self.style_layers) do
        styleLoss = styleLoss + mod.loss
    end
    for _, mod in ipairs(self.content_layers) do
        contentLoss = contentLoss + mod.loss
    end
    self.styleLoss = styleLoss
    self.contentLoss = contentLoss
    self.output = styleLoss+contentLoss
    return self.output
end

function criterion:updateGradInput(input, targets)
    if self.recompute_gradInput then
        local dy = self.dy:typeAs(self.net.output):resizeAs(self.net.output):zero()
        local grad = self.net:backward(input, dy)
        self.gradInput = grad:clone()
        -- reset targets
        if not self.targets then self:unsetTargets() end
    end
    self.recompute_gradInput = false
    return self.gradInput
end
