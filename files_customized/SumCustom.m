classdef SumCustom < dagnn.ElementWise
  %SumCustom DagNN sum layer
  %   The SumCustom layer takes inputs 
  %   and store the result = inputs{1}-inputs{2}
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = 2;
      outputs{1} = inputs{1} - input{2} ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = derOutputs{1} ;
      derInputs{2} = -derOutputs{1} ;
      
      derParams = {} ;
    end

    
% to check if dimensions of input layers are same:    
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('SumCustom layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = SumCustom(varargin)
      obj.load(varargin) ;
    end
  end
end
