require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
require 'paths'
require 'optim'
require 'lib/ArtisticStyleLossCriterion'
require 'lib/ImageLoaderAsync'
require 'lib/TVLossModule'
require 'lib/StyleINLossModule'
require 'lib/AdaptiveInstanceNormalization';
require 'lib/utils'

local cmd = torch.CmdLine()
-- Basic options
cmd:option('-contentDir', 'data/coco/train2014', 'Directory containing content images for training')
cmd:option('-styleDir', 'data/wikiart/train', 'Directory containing style images for training')
cmd:option('-name', 'adain', 'Name of the checkpoint directory')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use')
cmd:option('-nThreads', 2, 'Number of data loading threads')
cmd:option('-activation', 'relu', 'Activation function in the decoder')

-- Preprocessing options
cmd:option('-finalSize', 256, 'Size of images used for training')
cmd:option('-contentSize', 512, 'Size of content images before cropping, keep original size if set to 0')
cmd:option('-styleSize', 512, 'Size of style images before cropping, keep original size if set to 0')
cmd:option('-crop', true, 'If true, crop training images')

-- Training options
cmd:option('-resume', false, 'If true, resume training from the last checkpoint')
cmd:option('-optimizer', 'adam', 'Optimizer used, adam|sgd')
cmd:option('-learningRate', 1e-4, 'Learning rate')
cmd:option('-learningRateDecay', 5e-5, 'Learning rate decay')
cmd:option('-momentum', 0.9, 'Momentum')
cmd:option('-weightDecay', 0, 'Weight decay')
cmd:option('-batchSize', 8, 'Batch size')
cmd:option('-maxIter', 160000, 'Maximum number of iterations')
cmd:option('-targetContentLayer', 'relu4_1', 'Target content layer used to compute the loss')
cmd:option('-targetStyleLayers', 'relu1_1,relu2_1,relu3_1,relu4_1', 'Target style layers used to compute the loss')
cmd:option('-tvWeight', 0, 'Weight of TV loss')
cmd:option('-styleWeight', 1e-2, 'Weight of style loss')
cmd:option('-contentWeight', 1, 'Weight of content loss')
cmd:option('-reconStyle', false, 'If true, the decoder is also trained to reconstruct style images')
cmd:option('-normalize', false, 'If true, gradients at the loss function are normalized')

-- Verbosity
cmd:option('-printDetails', true, 'If true, print style loss at individual layers')
cmd:option('-display', true, 'If true, display the training progress')
cmd:option('-displayAddr', '0.0.0.0', 'Display address')
cmd:option('-displayPort', 8000, 'Display port')
cmd:option('-displayEvery', 100, 'Display interval')
cmd:option('-saveEvery', 2000, 'Save interval')
cmd:option('-printEvery', 10, 'Print interval')
opt = cmd:parse(arg)
opt.save = 'experiments/' .. opt.name
cutorch.setDevice(opt.gpu+1)
print(opt)

---- Prepare ----
if opt.contentSize == 0 then
    opt.contentSize = nil
end
if opt.styleSize == 0 then
    opt.styleSize = nil
end

assert(paths.dirp(opt.contentDir),
    '-contentDir does not exist.')
assert(paths.dirp(opt.styleDir),
    '-styleDir does not exist.')

if opt.display then
    display = require 'display'
    display.configure({hostname=opt.displayAddr, port=opt.displayPort})
end

if not opt.resume then
    paths.mkdir(opt.save)
    torch.save(paths.concat(opt.save, 'options.t7'), opt)
end

local decoderActivation
if opt.activation == 'relu' then
    decoderActivation = nn.ReLU
elseif opt.activation == 'prelu' then
    decoderActivation = nn.PReLU
elseif opt.activation == 'elu' then
    decoderActivation = nn.ELU
else
    error('Unknown activation option ' .. opt.activation)
end

---- Load VGG ----
vgg = torch.load('models/vgg_normalised.t7')
enc = nn.Sequential()
for i=1,#vgg do
    local layer = vgg:get(i)
    enc:add(layer)
    local name = layer.name
    if name == opt.targetContentLayer then
        break
    end
end

---- Build AdaIN layer ----
if enc:get(#enc).nOutputPlane ~= nil then
    adain = nn.AdaptiveInstanceNormalization(enc:get(#enc).nOutputPlane) -- assume the last encoder layer is Conv
else
    adain = nn.AdaptiveInstanceNormalization(enc:get(#enc-1).nOutputPlane) -- assume the last encoder layer is ReLU
end

---- Build decoder ----
if opt.resume then
    history = torch.load(paths.concat(opt.save, 'history.t7'))
    local startIter = #history
    local loc = paths.concat(opt.save, string.format('dec-%06d.t7', startIter))
    print("Resume training from: " .. loc)
    dec = torch.load(loc)
else
    history = {}
    dec = nn.Sequential()
    for i=#enc,1,-1 do
        local layer = enc:get(i)
        if torch.type(layer):find('SpatialConvolution') then
            local nInputPlane, nOutputPlane = layer.nOutputPlane, layer.nInputPlane
            dec:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
            dec:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1))
            dec:add(decoderActivation())
        end
        if torch.type(layer):find('MaxPooling') then
            dec:add(nn.SpatialUpSamplingNearest(2))
        end
    end
    dec:remove()
    dec:remove()
    dec:remove()
    dec:remove()
end

---- Build encoder ----
local layers = {}
layers.content = {opt.targetContentLayer}
layers.style = opt.targetStyleLayers:split(',')
local weights = {}
weights.content = opt.contentWeight
weights.style  = opt.styleWeight
weights.tv = opt.tvWeight
criterion = nn.ArtisticStyleLossCriterion(enc, layers, weights, opt.normalize)

---- Move to GPU ----
criterion.net = cudnn.convert(criterion.net, cudnn):cuda()
adain = adain:cuda()
dec = cudnn.convert(dec, cudnn):cuda()

print("encoder:")
print(criterion.net)
print("decoder:")
print(dec)

---- Build data loader ----
contentLoader = ImageLoaderAsync(opt.contentDir, opt.batchSize, {len=opt.contentSize, H=opt.finalSize, W=opt.finalSize, n=opt.nThreads}, opt.crop)
styleLoader = ImageLoaderAsync(opt.styleDir, opt.batchSize, {len=opt.styleSize, H=opt.finalSize, W=opt.finalSize, n=opt.nThreads}, opt.crop)
print("Number of content images: " .. contentLoader:size())
print("Number of style images: " .. styleLoader:size())

---- Training -----
if opt.resume then
    optimState = torch.load(paths.concat(opt.save, 'optimState.t7'))
else
    optimState = {
        learningRate = opt.learningRate,
        learningRateDecay = opt.learningRateDecay,
        weightDecay = opt.weightDecay,
        beta1 = opt.momentum,
        momentum = opt.momentum
    }
end

function maybe_print(trainLoss, contentLoss, styleLoss, tvLoss, timer)
    if optimState.iterCounter % opt.printEvery == 0 then
        print(string.format('%7d\t\t%e\t%e\t%e\t%e\t%.2f\t%e',
        optimState.iterCounter, trainLoss, contentLoss, styleLoss, tvLoss,
            timer:time().real, optimState.learningRate / (1 + optimState.iterCounter*optimState.learningRateDecay)))
        local allStyleLoss = {}
        for _, mod in ipairs(criterion.style_layers) do
            table.insert(allStyleLoss, mod.loss)
        end
        if opt.printDetails then
            print(allStyleLoss)
        end
        timer:reset()
    end
end

function maybe_display(inputs, reconstructions, history)
    if opt.display and (optimState.iterCounter % opt.displayEvery == 0) then
        local disp = torch.cat(reconstructions:float(), inputs:float(), 1)
        displayWindow = 1
        if displayWindow then
            styleNamesDisplayed = {}
            for i=1,#styleNames do
                local stylename = styleNames[i]
                local temp = stylename:split('/')
                local tempname = temp[#temp]
                table.insert(styleNamesDisplayed, tempname)
            end
            display.image(disp, {win=displayWindow, max=1, min=0, nperrow=opt.batchSize, labels=styleNamesDisplayed})
        else
            displayWindow = display.image(disp, {max=1, min=0})
        end
        display.plot(history, {win=displayWindow+1, title="loss: " .. opt.name, 
            labels = {"iteration", "loss", "content", "style", 'style_recon'}})
    end
end

function maybe_save()
    if optimState.iterCounter % opt.saveEvery == 0 then
        local loc = paths.concat(opt.save, string.format('dec-%06d.t7', optimState.iterCounter))
        local decSaved = dec:clearState():clone()
        torch.save(loc, cudnn.convert(decSaved:float(), nn))
        torch.save(paths.concat(opt.save, 'history.t7'), history)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        dec:clearState()
        criterion.net:clearState()
        decSaved = nil
        collectgarbage()
    end
end

function train()
    optimState.iterCounter = optimState.iterCounter or 0
    local weights, gradients = dec:getParameters()
    print('Training...\tTrainErr\tContent\t\tStyle\t\tTVLoss\t\ttime\tLearningRate')
    local timer = torch.Timer()
    while optimState.iterCounter < opt.maxIter do
        function feval(x)
            gradients:zero()
            optimState.iterCounter = optimState.iterCounter + 1
            contentInput = contentLoader:nextBatch()
            styleInput, styleNames = styleLoader:nextBatch()
            contentInput = contentInput:float():cuda()
            styleInput = styleInput:float():cuda()

            -- Forward style image through the encoder
            criterion:setStyleTarget(styleInput)
            local styleLatent = criterion.net.output:clone()

            -- Forward content image through the encoder
            criterion:setContentTarget(contentInput)
            local contentLatent = criterion.net.output:clone()
            
            -- Perform AdaIN
            local outputLatent = adain:forward({contentLatent, styleLatent})

            -- Set content target
            criterion.content_layers[1]:setTarget(outputLatent)

            -- Compute loss
            output = dec:forward(outputLatent):clone() -- forward through decoder, generate transformed images
            local loss = criterion:forward(output) -- forward through loss network, compute loss functions
            local contentLoss = criterion.contentLoss
            local styleLoss = criterion.styleLoss
            local tvLoss = 0
            if opt.tvWeight > 0 then
                tvLoss = criterion.net:get(2).loss
            end

            -- Backpropagate gradients
            local decGrad = criterion:backward(output) -- backprop through loss network, compute gradients w.r.t. the transformed images
            dec:backward(outputLatent, decGrad) -- backprop gradients through decoder

            -- Optionally train the decoder to reconstruct style images
            local styleReconLoss = 0
            if opt.reconStyle then
                criterion:setContentTarget(styleInput)
                styleRecon = dec:forward(styleLatent):clone()
                styleReconLoss = criterion:forward(styleRecon)
                local decGrad = criterion:backward(styleRecon)
                dec:backward(styleLatent, decGrad)
                loss =  loss + styleReconLoss
            end
            
            table.insert(history, {optimState.iterCounter, loss, contentLoss, styleLoss, styleReconLoss})
            maybe_print(loss, contentLoss, styleLoss, tvLoss, timer)
            if opt.reconStyle then
                displayImages = torch.cat({output, styleRecon}, 1)
            else
                displayImages = output
            end
            criterion.net:clearState()
            maybe_display(torch.cat({contentInput, styleInput}, 1), displayImages, history)
            maybe_save()
            return loss, gradients
        end

        if opt.optimizer == 'adam' then
            optim.adam(feval, weights, optimState)
        elseif opt.optimizer == 'sgd' then
            optim.sgd(feval, weights, optimState)
        else
            error("Not supported optimizer: " .. opt.optimizer)
        end
    end
end

train()
