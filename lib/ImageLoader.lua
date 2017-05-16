require 'torch'
require 'paths'
require 'image'

local ImageLoader = torch.class('ImageLoader')

function ImageLoader:__init(dir)
    local files = paths.dir(dir)
    local i = 1
    while i <= #files do
        if not string.find(files[i], 'jpg$') 
            and not string.find(files[i], 'png$')
            and not string.find(files[i], 'ppm$')then
            table.remove(files, i)
        else
            i = i +1
        end
    end
    self.dir = dir
    self.files = files
    self:rebatch()
end

function ImageLoader:rebatch()
    self.perm = torch.randperm(#self.files)
    self.idx = 1
end

function ImageLoader:next()
    -- load image
    local img = nil
    local name
    while true do
        if self.idx > #self.files then self:rebatch() end
        local i = self.perm[self.idx]
        name = self.files[i]
        local loc = paths.concat(self.dir, name) 
        self.idx = self.idx + 1 
        local status,err = pcall(function() img = image.load(loc,3,'float') end)
        if status then 
            if self.verbose then
                print('Loaded ' .. self.files[i])
            end
            break 
        else
            io.stderr:write('WARN: Failed to load ' .. loc .. ' due to error: ' .. err .. '\n')
        end
    end
    
    -- preprocess
    local H, W = img:size(2), img:size(3)
    if self.len then
        img = image.scale(img, self.len)
    elseif self.max_len then
        if H > self.max_len or W > self.max_len then
            img = image.scale(img, self.max_len)
        end
    end
    
    H, W = img:size(2), img:size(3)
    if self.div then
        local Hc = math.floor(H / self.div) * self.div
        local Wc = math.floor(W / self.div) * self.div
        img = self:_randomCrop(img, Hc, Wc)
    end

    if self.bnw then
        img = image.rgb2yuv(img)
        img[2]:zero()
        img[3]:zero()
        img = image.yuv2rgb(img)
    end

    collectgarbage()
    return img, name
end

function ImageLoader:setVerbose(verbose)
    verbose = verbose or true
    self.verbose = verbose
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
    assert(oheight <= H)
    assert(owidth <= W)
    local y = torch.floor(torch.uniform(0, H-oheight+1))
    local x = torch.floor(torch.uniform(0, W-owidth+1))
    local crop_img = image.crop(img, x,y, x+owidth, y+oheight)
    return crop_img
end

function ImageLoader:setBlackNWhite(bool)
    if bool then
        self.bnw = true
    else
        self.bnw = false
    end
end
