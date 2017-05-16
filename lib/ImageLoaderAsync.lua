require 'torch'

local ImageLoaderAsync = torch.class('ImageLoaderAsync')

local threads = require 'threads'

local ImageLoader = {}
local ImageLoader_mt = { __index = ImageLoader }

---- Asynchronous image loader.
local result = {}
local H, W
local len

function ImageLoaderAsync:__init(dir, batchSize, options, crop)
    if not batchSize then
        error('Predetermined batch size is required for asynchronous loader.')
    end
    options = options or {}
    local n = options.n or 1

    -- upvalues
    H,W = options.H, options.W
    len = options.len

    self.batchSize = batchSize
    self._type = 'torch.FloatTensor'

    -- initialize thread and its image loader
    self.threads = threads.Threads(n,
    function()
        imageLoader = ImageLoader:new(dir)
        if H ~= nil and W ~= nil then
            imageLoader:setWidthAndHeight(W,H)
        end
        if len ~= nil then
            imageLoader:setFitToHeightOrWidth(len)
        end
        imageLoader.crop = crop
    end)

    -- get size
    self.threads:addjob(
    function() return imageLoader:size() end,
    function(size) result[1] = size end)
    self.threads:dojob()
    self._size = result[1]
    result[1] = nil
    result[2] = nil
    result[3] = nil

    -- add job
    for i=1,n do
        self.threads:addjob(self.__getBatchFromThread, self.__pushResult, self.batchSize)
    end
end

function ImageLoaderAsync:size()
    return self._size
end

function ImageLoaderAsync:type(type)
    if not type then
        return self._type
    else
        assert(torch.Tensor():type(type), 'Invalid type ' .. type .. '?')
        self._type = type
    end
    return self
end

function ImageLoaderAsync.__getBatchFromThread(batchSize)
    a,b,c = imageLoader:nextBatch(batchSize)
    return a,b,c
end

function ImageLoaderAsync.__pushResult(batch, names)
    result[1] = batch
    result[2] = names
end

function ImageLoaderAsync:nextBatch()
    self.threads:addjob(self.__getBatchFromThread, self.__pushResult, self.batchSize)
    self.threads:dojob()
    local batch = result[1]
    result[1] = nil
    local names = result[2]
    result[2] = nil
    return batch:type(self._type), names
end

---- Implementation of the actual image loader.
function ImageLoader:new(dir)
    require 'torch'
    require 'paths'
    require 'image'
    require 'lib/utils'

    local imageLoader = {}
    setmetatable(imageLoader, ImageLoader_mt)
    files = extractImageNamesRecursive(dir)
    imageLoader.dir = dir
    imageLoader.files = files
    imageLoader.tm = torch.Timer()
    imageLoader.tm:reset()
    imageLoader:rebatch()
    return imageLoader
end

function ImageLoader:size()
    return #self.files
end

function ImageLoader:rebatch()
    self.perm = torch.randperm(self:size())
    self.idx = 1
end

function ImageLoader:nextBatch(batchSize)
    local img, name = self:next()
    local batch = torch.FloatTensor(batchSize, 3, img:size(2), img:size(3))
    local names = {}
    batch[1] = img
    table.insert(names, name)
    for i=2,batchSize do
        local temp, tempname = self:next()
        batch[i] = temp
        table.insert(names, tempname)
    end
    return batch, names
end

function ImageLoader:next()
    -- load image
    local img = nil
    local name
    local numErr = 0
    while true do
        if self.idx > self:size() then self:rebatch() end
        local i = self.perm[self.idx]
        self.idx = self.idx + 1
        name = self.files[i]
        local loc = paths.concat(self.dir, name)
        local status,err = pcall(function() img = image.load(loc,3,'float') end) -- load in range (0,1)
        if status then
            if self.verbose then print('Loaded ' .. self.files[i]) end
            break
        else
            io.stderr:write('WARNING: Failed to load ' .. loc .. ' due to error: ' .. err .. '\n')
        end
    end

    -- preprocess
    local H, W = img:size(2), img:size(3)

    if self.len ~= nil then -- resize without changing aspect ratio
        img = image.scale(img, "^" .. self.len)
    end

    if self.crop then
        img = self:_randomCrop(img, self.H, self.W)
    else
        if self.W and self.H then -- resize
            img = image.scale(img, self.W, self.H) 
        elseif self.max_len then -- resize without changing aspect ratio
            if H > self.max_len or W > self.max_len then
                img = image.scale(img, self.max_len)
            end
        end
    end

    collectgarbage()
    return img, name, numErr
end

---- Optional preprocessing
function ImageLoader:setVerbose(verbose)
    verbose = verbose or true
    self.verbose = verbose
end

function ImageLoader:setWidthAndHeight(W,H)
    self.H = H
    self.W = W
end

function ImageLoader:setFitToHeightOrWidth(len)
    assert(len ~= nil)
    self.len = len
    self.max_len = nil
end

function ImageLoader:setMaximumSize(max_len)
    assert(max_len ~= nil)
    self.max_len = max_len
    self.len = nil
end

function ImageLoader:setDivisibleBy(div)
    assert(div ~= nil)
    self.div = div
end

function ImageLoader:_randomCrop(img, oheight, owidth)
    assert(img:dim()==3)
    local H,W = img:size(2), img:size(3)
    if oheight > H then
        print(oheight, H)
        error()
    end
    if owidth > W then
        print(owidth, W)
        error()
    end
    assert(oheight <= H)
    assert(owidth <= W)
    local y = torch.floor(torch.uniform(0, H-oheight+1))
    local x = torch.floor(torch.uniform(0, W-owidth+1))
    local crop_img = image.crop(img, x,y, x+owidth, y+oheight)
    return crop_img
end
