require 'torch'
require 'unsup'
require 'nn'
require 'image'
require 'paths'
require 'lib/AdaptiveInstanceNormalization'
require 'lib/utils'
require 'riseml'

local opt
local vgg, adain, decoder
local styleFeatures = {}

local function getStyleFeature(style)
    if opt.gpu >= 0 then
        style = style:cuda()
    else
        style = style:float()
    end    
    return vgg:forward(style):clone()
end

local function init()

    local cmd = torch.CmdLine()

    -- Basic options
    cmd:option('-style', 'asheville', 'Style name')
    cmd:option('-styleExt', 'jpg', 'The extension name of the style image')
    cmd:option('-styleDir', '/code/input/style', 'Directory path to style images')
    cmd:option('-vgg', 'models/vgg_normalised.t7', 'Path to the VGG network')
    cmd:option('-decoder', 'models/decoder-content-similar.t7', 'Path to the decoder')

    -- Additional options
    cmd:option('-contentSize', 512,
               'New (minimum) size for the content image, keeping the original size if set to 0')
    cmd:option('-styleSize', 512,
               'New (minimum) size for the style image, keeping the original size if set to 0')
    cmd:option('-crop', false, 'If true, center crop both content and style image before resizing')
    cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

    -- Advanced options
    cmd:option('-alpha', 1, 'The weight that controls the degree of stylization. Should be between 0 and 1')

    opt = cmd:parse(arg)
    
    local gpu = tonumber(os.getenv("GPU"))
    if gpu ~= nil then
        opt.gpu = gpu
    end    

    print(opt)
    if opt.gpu >= 0 then
        require 'cudnn'
        require 'cunn'
    end

    assert(opt.style ~= '', '--style should be given.')
    assert(paths.filep(opt.decoder), 'Decoder ' .. opt.decoder .. ' does not exist.')

    vgg = torch.load(opt.vgg)
    for i=53,32,-1 do
        vgg:remove(i)
    end
    adain = nn.AdaptiveInstanceNormalization(vgg:get(#vgg-1).nOutputPlane)
    decoder = torch.load(opt.decoder)
  
    if opt.gpu >= 0 then
        cutorch.setDevice(opt.gpu+1)
        vgg = cudnn.convert(vgg, cudnn):cuda()
        -- vgg:cuda()
        adain:cuda()
        -- decoder = cudnn.convert(decoder, cudnn):cuda()
        decoder:cuda()
    else
        vgg:float()
        adain:float()
        decoder:float()
    end    
end

local function styleTransfer(content, styleFeature)

    if opt.gpu >= 0 then
        content = content:cuda()
    else
        content = content:float()
    end

    local contentFeature = vgg:forward(content):clone()
    local targetFeature = adain:forward({contentFeature, styleFeature})

    targetFeature = targetFeature:squeeze()
    targetFeature = opt.alpha * targetFeature + (1 - opt.alpha) * contentFeature
    return decoder:forward(targetFeature) 
end

local function process(contentImgData)
    local contentByteTensor = torch.ByteTensor(torch.ByteStorage():string(contentImgData))
    local contentImg = image.decompressJPG(contentByteTensor, 3)
    contentImg = sizePreprocess(contentImg, opt.crop, opt.contentSize)

    local style = opt.style -- need to be passed as a parameter from server

    if styleFeatures[opt.style] == nil then
        local stylePath = paths.concat(opt.styleDir, opt.style .. '.' .. opt.styleExt)        
        if not paths.filep(stylePath) then
            print("Style " .. stylePath .. " not found")
        else                
            print("Calculate style feature: " .. stylePath)
            local styleImg = image.load(stylePath, 3, 'float')
            styleImg = sizePreprocess(styleImg, opt.crop, opt.styleSize)
            local styleFeature = getStyleFeature(styleImg)
            styleFeatures[opt.style] = styleFeature
        end
    else
        print("Use pre-calculated style feature")
    end

    local outputData = contentImgData
    if styleFeatures[opt.style] ~= nil then
        local output = styleTransfer(contentImg, styleFeatures[opt.style])
        outputData = image.compressJPG(output):storage():string()
    end
    return outputData
end

local function main()
    init()
    riseml.serve(process)
end

main()
