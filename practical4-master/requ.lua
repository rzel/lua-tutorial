require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  mask = torch.gt(self.output, 0)
  self.output:maskedFill(mask, 0)
  self.output = torch.cmul(self.output, self.output)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here...
  self.output:resizeAs(input):copy(input)
  mask = torch.gt(self.output, 0)
  self.output:maskedFill(mask, 0)
  self.gradInput:cmul(self.output)
  self.gradInput:mul(2) 

  return self.gradInput
end

