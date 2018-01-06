require "indie-classes"


indie = indie or {}

indie.ml = {}
local ml = indie.ml

--NeuralNetwork-------------------------------------------------

ml.NeuralNetwork = indie.classes.createClass()

-- constructor // static methods
function indie.ml.NeuralNetwork.new(inputSize, outputSize)
  
  assert(type(inputSize) == "number", "Argument #1: Expected number, got "..type(inputSize))
  assert(type(outputSize) == "number", "Argument #2: Expected number, got "..type(outputSize))
  
  local instance = {
    inputNodes = {},
    hiddenNodes = {},
    outputNodes = {}
  }
  
  for i = 1, inputSize do
    table.insert(instance.inputNodes, ml.InputNode.new())
  end
  for i = 1, outputSize do
    table.insert(instance.outputNodes, ml.AggregateNode.new())
  end
  
  return setmetatable(instance, ml.NeuralNetwork)
end

-- methods

function ml.NeuralNetwork:getInputSize()
  return #self.inputNodes
end

function ml.NeuralNetwork:getOutputSize()
  return #self.outputNodes
end

function ml.NeuralNetwork:getInputNode(index)
  return self.inputNodes[index]
end

function ml.NeuralNetwork:getOutputNode(index)
  return self.outputNodes[index]
end

function ml.NeuralNetwork:getHiddenNode(index)
  return self.hiddenNodes[index]
end

function ml.NeuralNetwork:addInputNode(inputNode)
  assert(type(inputNode) == "table", "Argument #1: Expected table, got "..type(inputNode))
  assert(inputNode.is_a and inputNode.is_a[ml.InputNode], "Argument #1: Type does not match ml.InputNode")
  
  for k,v in ipairs(self.inputNodes) do
    assert(v ~= inputNode, "Cannot add identical input node twice in the same NeuralNetwork")
  end
  
  table.insert(self.inputNodes, inputNode)
end

function ml.NeuralNetwork:addOutputNode(outputNode)
  assert(type(outputNode) == "table", "Argument #1: Expected table, got "..type(outputNode))
  assert(outputNode.is_a and outputNode.is_a[ml.AggregateNode], "Argument #1: Type does not match ml.AggregateNode")
  
  for k,v in ipairs(self.outputNodes) do
    assert(v ~= outputNode, "Cannot add identical output node twice in the same NeuralNetwork")
  end
  
  table.insert(self.outputNodes, outputNode)
end

function ml.NeuralNetwork:addHiddenNode(hiddenNode)
  assert(type(hiddenNode) == "table", "Argument #1: Expected table, got "..type(outputNode))
  assert(hiddenNode.is_a and hiddenNode.is_a[ml.AggregateNode], "Argument #1: Type does not match ml.AggregateNode")
  
  for k,v in ipairs(self.outputNodes) do
    assert(v ~= hiddenNode, "Cannot add identical hidden node twice in the same NeuralNetwork")
  end
  
  table.insert(self.hiddenNodes, hiddenNode)
end

function ml.NeuralNetwork:removeInputNode(inputNode)
  
  assert(type(inputNode) == "table", "Argument #1: Expected table, got "..type(inputNode))
  assert(inputNode.is_a and inputNode.is_a[ml.InputNode], "Argument #1: Type does not match ml.InputNode")
  
  for k,v in pairs(self.inputNodes) do
    if (v == inputNode) then
      table.remove(self.inputNodes, k)
      break
    end
  end
end

function ml.NeuralNetwork:removeOutputNode(outputNode)
  
  assert(type(outputNode) == "table", "Argument #1: Expected table, got "..type(outputNode))
  assert(outputNode.is_a and outputNode.is_a[ml.AggregateNode], "Argument #1: Type does not match ml.AggregateNode")
  
  for k,v in pairs(self.outputNodes) do
    if (v == outputNode) then
      table.remove(self.outputNodes, k)
      break
    end
  end
end

function ml.NeuralNetwork:removeHiddenNode(hiddenNode)
  
  assert(type(hiddenNode) == "table", "Argument #1: Expected table, got "..type(hiddenNode))
  assert(hiddenNode.is_a and hiddenNode.is_a[ml.AggregateNode], "Argument #1: Type does not match ml.AggregateNode")
  
  for k,v in pairs(self.hiddenNodes) do
    if (v == hiddenNode) then
      table.remove(self.hiddenNodes, k)
      break
    end
  end
end

function ml.NeuralNetwork:evaluate(...)
  local inputValues = {...}
  
  assert(#inputValues == #self.inputNodes, "Number of arguments("..tostring(#inputValues)..") does not match number of input nodes("..tostring(#self.inputNodes)..")")
  
  self:reset()
  
  for k,v in pairs(self.inputNodes) do
    assert(type(inputValues[k]) == "number", "Argument #"..tostring(k)..": Expected number, got "..type(inputValues[k]))
    v:setValue(inputValues[k])
  end
  
  local outputValues = {}
  
  for k,v in pairs(self.outputNodes) do
    table.insert(outputValues, v:evaluate())
  end
  
  return unpack(outputValues)
end

function ml.NeuralNetwork:reset()
  for k,v in pairs(self.inputNodes) do
    v:reset()
  end
  
  for k,v in pairs(self.hiddenNodes) do
    v:reset()
  end
  
  for k,v in pairs(self.outputNodes) do
    v:reset()
  end
end


--NeuralNode----------------------------------------------------

ml.NeuralNode = indie.classes.createClass()

function ml.NeuralNode.new()
  local instance = {
    value = nil,
    recurrentValue = nil
  }
  
  return setmetatable(instance, ml.NeuralNode)
end

function ml.NeuralNode:reset()
  self.recurrentValue = self.value
  self.value = nil
end

function ml.NeuralNode:evaluate()
  assert(self.value ~= nil, "Unable to evaluate NeuralNode without value")
  
  return self.value
end

function ml.NeuralNode:evaluateRecurrent()
  return self.recurrectValue or 0
end

--InputNode-----------------------------------------------------

ml.InputNode = indie.classes.createClass(ml.NeuralNode)

function ml.InputNode.new()
  local instance = ml.NeuralNode.new()
  
  return setmetatable(instance, ml.InputNode)
end

function ml.InputNode:setValue(value)
  assert(type(value) == "number", "Argument #1: Expected number, got "..type(value))
  
  self.value = value
end

--AggregateNode-------------------------------------------------

ml.AggregateNode = indie.classes.createClass(ml.InputNode)

function ml.AggregateNode.new()
  local instance = ml.NeuralNode.new()
  
  instance.inputs = {}
  instance.recurrentInputs = {}
  
  return setmetatable(instance, ml.AggregateNode)
end

function ml.AggregateNode:setIncomingConnection(incomingNode, incomingWeight)
  assert(type(incomingNode) == "table", "Argument #1: Expected table, got "..type(incomingNode))
  assert(type(incomingWeight) == "number", "Argument #2: Expected number, got "..type(incomingWeight))
  assert(incomingNode.is_a and incomingNode.is_a[ml.NeuralNode], "Argument #1: Type does not match ml.NeuralNode")
  
  if (incomingWeight == 0) then incomingWeight = nil end
  
  self.inputs[incomingNode] = incomingWeight
end

function ml.AggregateNode:setIncomingRecurrentConnection(incomingNode, incomingWeight)
  assert(type(incomingNode) == "table", "Argument #1: Expected table, got "..type(incomingNode))
  assert(type(incomingWeight) == "number", "Argument #2: Expected number, got "..type(incomingWeight))
  assert(incomingNode.is_a and incomingNode.is_a[ml.NeuralNode], "Argument #1: Type does not match ml.NeuralNode")
  
  if (incomingWeight == 0) then incomingWeight = nil end
  
  self.recurrentInputs[incomingNode] = incomingWeight
end

function ml.AggregateNode:evaluate()
  if (self.value == nil) then
    local val = 0
    
    for inputNode,inputWeight in pairs(self.inputs) do
      val = val + inputNode:evaluate() * inputWeight
    end
    
    for inputNode,inputWeight in pairs(self.recurrentInputs) do
      val = val + inputNode:evaluateRecurrent() * inputWeight
    end
    
    -- transfer function
    self.value = ml.TransferFunctions.Sigmoid(val)
  end

  return self.value
end

ml.TransferFunctions = {
  Sigmoid = function (v)
    return 1 / (1 + math.exp(-v))
  end
}

-- Love2d network visualizer extensions
if (love and love.graphics) then
  
  local internalDefaults = {
    width = 190,
    height = 190,
    x = 10,
    y = 10
  }
  
  function ml.NeuralNetwork:setSize(width, height)
    assert(type(width) == "number", "Argument #1: Expected number, got "..type(width))
    assert(type(height) == "number", "Argument #2: Expected number, got "..type(height))
    
    assert(width >= 0, "Argument #1: Value must be positive")
    assert(height >= 0, "Argument #2: Value must be positive")
    
    self.love = self.love or {}
    
    self.love.width = width
    self.love.height = height
  end
  function ml.NeuralNetwork:setPosition(x, y)
    
    assert(type(x) == "number", "Argument #1: Expected number, got "..type(x))
    assert(type(y) == "number", "Argument #2: Expected number, got "..type(y))
    
    self.love = self.love or {}
    
    self.love.x = x
    self.love.y = y
  end
  function ml.NeuralNetwork:getSize()
     return (self.love or internalDefaults).width or internalDefaults.width,
      (self.love or internalDefaults).height or internalDefaults.height
  end
  function ml.NeuralNetwork:getPosition()
    return (self.love or internalDefaults).x or internalDefaults.x,
      (self.love or internalDefaults).y or internalDefaults.y
  end
  function ml.NeuralNetwork:autoLayout()
    
    self.love = self.love or {}
    self.love.nodePositions = {}
    
    local width, height = self:getSize()
    local x,y = self:getPosition()
    
    for i,v in ipairs(self.inputNodes) do
      self.love.nodePositions[v] = {
        x = x + width / (#self.inputNodes+1) * i,
        y = y
      }
    end
    
    for i,v in ipairs(self.outputNodes) do
      self.love.nodePositions[v] = {
        x = x + width / (#self.outputNodes+1) * i,
        y = y + height
      }
    end
    
    return self.love.nodePositions
  end
  function ml.NeuralNetwork:draw()
    
    self.love = self.love or {}
    self.love.nodePositions = self.love.nodePositions or self:autoLayout()
    
    -- draw connections
    
    -- draw nodes
    love.graphics.setColor(0,0,0)
    for k,v in pairs(self.inputNodes) do
      love.graphics.circle("fill", self.love.nodePositions[v].x, self.love.nodePositions[v].y, 5)
    end
    for k,v in pairs(self.outputNodes) do
      love.graphics.circle("fill", self.love.nodePositions[v].x, self.love.nodePositions[v].y, 5)
    end
  end
end
